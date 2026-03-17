import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

import puppeteer from "puppeteer-core";
import YAML from "yaml";

const DEFAULT_INPUTS = [
  "03-methodology.qmd",
  "03-modelling-decision-function.qmd",
  "appendix-decision-architecture.qmd",
];

function parseQuartoMermaidBlock(rawBlock) {
  const lines = rawBlock.split(/\r?\n/);

  let label;
  const nonDirectiveLines = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("%%|")) {
      const m = trimmed.match(/^%%\|\s*label:\s*(.+?)\s*$/);
      if (m) label = m[1];
      continue;
    }
    nonDirectiveLines.push(line);
  }

  const cleaned = nonDirectiveLines.join("\n").trimStart();

  // Quarto allows a YAML block at the top of the mermaid code fence:
  // ---
  // config: ...
  // ---
  // <actual mermaid>
  if (!cleaned.startsWith("---")) {
    return {
      label,
      mermaidConfig: undefined,
      diagram: cleaned.trim(),
    };
  }

  const yamlEnd = cleaned.indexOf("\n---", 3);
  if (yamlEnd === -1) {
    // Treat as raw diagram if malformed.
    return {
      label,
      mermaidConfig: undefined,
      diagram: cleaned.trim(),
    };
  }

  const yamlText = cleaned.slice(3, yamlEnd).trim();
  const after = cleaned.slice(yamlEnd + "\n---".length).trim();

  let parsed;
  try {
    parsed = YAML.parse(yamlText);
  } catch (err) {
    throw new Error(`Failed to parse mermaid YAML config: ${err?.message ?? err}`);
  }

  const mermaidConfig = parsed?.config;

  return {
    label,
    mermaidConfig,
    diagram: after,
  };
}

async function extractMermaidBlocksFromQmd(qmdPath) {
  const text = await fs.readFile(qmdPath, "utf8");
  const blocks = [];

  const re = /```\{mermaid\}\s*\n([\s\S]*?)\n```/g;
  let match;
  while ((match = re.exec(text)) !== null) {
    blocks.push(match[1]);
  }
  return blocks;
}

async function renderMermaidToAssets({
  browser,
  diagram,
  mermaidConfig,
  label,
  pngPath,
  pdfPath,
}) {
  const require = createRequire(import.meta.url);
  const mermaidEsmPath = require.resolve("mermaid/dist/mermaid.esm.min.mjs");
  const elkEsmPath = require.resolve(
    "@mermaid-js/layout-elk/dist/mermaid-layout-elk.esm.min.mjs",
  );
  const bootstrapIconsUrl =
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css";

  const mermaidUrl = pathToFileURL(mermaidEsmPath).toString();
  const elkUrl = pathToFileURL(elkEsmPath).toString();

  const config = {
    startOnLoad: false,
    securityLevel: "loose",
    ...(mermaidConfig ?? {}),
  };

  const id = label ? `m_${label}` : `m_${Math.random().toString(16).slice(2)}`;

  const figureId = label ?? "mermaid-figure";
  const html = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="${bootstrapIconsUrl}" />
    <style>
      body { margin: 0; padding: 0; }
    </style>
  </head>
  <body>
    <div id="${figureId}">
      <div id="out" class="mermaid"></div>
    </div>
    <script type="module">
      import mermaid from ${JSON.stringify(mermaidUrl)};
      import elkLayouts from ${JSON.stringify(elkUrl)};

      if (mermaid?.registerLayoutLoaders) {
        mermaid.registerLayoutLoaders(elkLayouts);
      }

      mermaid.initialize(${JSON.stringify(config)});
      const diagram = ${JSON.stringify(diagram)};
      try {
        const { svg } = await mermaid.render(${JSON.stringify(id)}, diagram);
        document.getElementById('out').innerHTML = svg;
        if (document.fonts?.ready) {
          await document.fonts.ready;
        }
        // Signal completion for puppeteer waiters.
        window.__MERMAID_DONE__ = true;
      } catch (err) {
        window.__MERMAID_ERROR__ = String(err?.stack ?? err);
      }
    </script>
  </body>
</html>`;

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });

  const tmpHtml = path.join(
    path.dirname(pngPath),
    `.__tmp_render_${label ?? "mermaid"}.html`,
  );
  await fs.mkdir(path.dirname(tmpHtml), { recursive: true });
  await fs.writeFile(tmpHtml, html, "utf8");

  page.on("pageerror", (err) => {
    // eslint-disable-next-line no-console
    console.error("[mermaid prerender] pageerror:", err);
  });
  page.on("console", (msg) => {
    // eslint-disable-next-line no-console
    console.log("[mermaid prerender]", msg.type(), msg.text());
  });

  await page.goto(pathToFileURL(tmpHtml).toString(), { waitUntil: "load" });
  await page.waitForFunction(
    "window.__MERMAID_DONE__ === true || !!window.__MERMAID_ERROR__",
    { timeout: 60_000 },
  );

  const errorText = await page.evaluate(() => window.__MERMAID_ERROR__ ?? null);
  if (errorText) {
    await page.close();
    await fs.unlink(tmpHtml).catch(() => undefined);
    throw new Error(`Mermaid render failed: ${errorText}`);
  }

  const svgHandle = await page.$("svg");
  if (!svgHandle) {
    await page.close();
    await fs.unlink(tmpHtml).catch(() => undefined);
    throw new Error("Mermaid render failed: SVG element not found");
  }

  // Rasterize to PNG so PDF/LaTeX doesn't need rsvg-convert.
  await svgHandle.screenshot({ path: pngPath, omitBackground: true });

  const bbox = await page.evaluate(() => {
    const svg = document.querySelector("svg");
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    return {
      width: Math.ceil(rect.width),
      height: Math.ceil(rect.height),
    };
  });
  if (!bbox || !Number.isFinite(bbox.width) || !Number.isFinite(bbox.height)) {
    await page.close();
    await fs.unlink(tmpHtml).catch(() => undefined);
    throw new Error("Mermaid render failed: unable to measure SVG bounds");
  }

  await page.setViewport({
    width: Math.max(1, bbox.width),
    height: Math.max(1, bbox.height),
    deviceScaleFactor: 2,
  });
  await page.evaluate(() => {
    document.body.style.margin = "0";
    document.body.style.padding = "0";
    const out = document.getElementById("out");
    if (out) {
      out.style.display = "inline-block";
      out.style.margin = "0";
      out.style.padding = "0";
    }
  });
  await page.pdf({
    path: pdfPath,
    printBackground: true,
    width: `${bbox.width}px`,
    height: `${bbox.height}px`,
    margin: { top: "0px", right: "0px", bottom: "0px", left: "0px" },
    preferCSSPageSize: true,
  });
  await page.close();

  await fs.unlink(tmpHtml).catch(() => undefined);

  // Note: assets are written by puppeteer screenshot/pdf.
}

async function main() {
  // Skip on preview/incremental renders to keep it snappy.
  if (!process.env.QUARTO_PROJECT_RENDER_ALL) {
    process.exit(0);
  }

  const chromiumPath =
    process.env.PUPPETEER_EXECUTABLE_PATH ??
    process.env.CHROMIUM_BIN ??
    "/usr/bin/chromium-browser";

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: chromiumPath,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--allow-file-access-from-files",
      "--disable-web-security",
    ],
  });

  const inputs = (process.env.LLM_ABM_MERMAID_INPUTS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const qmdFiles = inputs.length ? inputs : DEFAULT_INPUTS;

  const generatedDir = path.join("_generated", "mermaid");
  const rendered = [];
  const seenLabels = new Set();

  try {
    for (const qmdFile of qmdFiles) {
      const blocks = await extractMermaidBlocksFromQmd(qmdFile);
      for (const block of blocks) {
        const { label, mermaidConfig, diagram } = parseQuartoMermaidBlock(block);
        if (!label) {
          // Only render blocks that have stable, referenced labels.
          continue;
        }

        if (seenLabels.has(label)) {
          continue;
        }
        seenLabels.add(label);

        const pngPath = path.join(generatedDir, `${label}.png`);
        const pdfPath = path.join(generatedDir, `${label}.pdf`);
        await renderMermaidToAssets({
          browser,
          diagram,
          mermaidConfig,
          label,
          pngPath,
          pdfPath,
        });
        rendered.push(pngPath, pdfPath);
      }
    }
  } finally {
    await browser.close();
  }

  if (rendered.length) {
    console.log(`Pre-rendered Mermaid diagrams (${rendered.length}):`);
    for (const file of rendered) console.log(`- ${file}`);
  }
}

await main();
