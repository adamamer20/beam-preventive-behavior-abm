"""Prompt labeling and semantics helpers."""

from __future__ import annotations


class PromptLabelsMixin:
    def _display_label(self, col: str) -> str:
        return self.display_labels.get(col) or self.clean_questions.get(col) or col

    def outcome_label(self, outcome: str) -> str:
        label = self._display_label(outcome)
        bounds = self.possible_bounds.get(outcome)
        endpoints = self.ordinal_endpoints.get(outcome)
        if bounds is not None:
            lo, hi = bounds
            lo_s, hi_s = self.fmt_num(float(lo)), self.fmt_num(float(hi))
            if endpoints is not None:
                lo_lab, hi_lab = endpoints
                return f"{label} ({lo_s}–{hi_s}: {lo_s} = {lo_lab}, {hi_s} = {hi_lab})"
            return f"{label} ({lo_s}–{hi_s})"
        return label

    def outcome_semantics(self, outcome: str | None) -> tuple[str, str]:
        if not outcome:
            return "Outcome semantics: (unknown)", 'Return ONLY JSON: {"prediction": <float>}.'
        label = self.outcome_label(outcome)
        outcome_type = str(self.column_types.get(outcome, "")).strip().lower()
        if outcome_type == "binary":
            schema = 'Return ONLY JSON: {"prediction": <float in [0,1]>}.'
        else:
            schema = 'Return ONLY JSON: {"prediction": <float>}. Use the stated scale bounds.'
        return f"Outcome semantics: {label}", schema

    def multi_outcome_semantics(self, outcomes: list[str]) -> tuple[str, str]:
        lines = ["Outcome semantics:"]
        for outcome in outcomes:
            label = self.outcome_label(outcome)
            t = str(self.column_types.get(outcome, "")).strip().lower()
            if t == "binary":
                label = f"{label} (binary probability in [0,1])"
            lines.append(f"- {outcome}: {label}")
        schema = "Return ONLY JSON with keys: " + ", ".join(outcomes) + "."
        schema += " Each value must stay within the stated scale bounds."
        return "\n".join(lines), schema

    def psychometrician_system(self, outcome: str | None) -> str:
        semantics, schema = self.outcome_semantics(outcome)
        return (
            "You are an expert survey psychometrician acting as a prediction function.\n\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n\n"
            f"{semantics}\n\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def psychometrician_system_multi(self, outcomes: list[str]) -> str:
        semantics, schema = self.multi_outcome_semantics(outcomes)
        return (
            "You are an expert survey psychometrician acting as a prediction function.\n\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n\n"
            f"{semantics}\n\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    @staticmethod
    def first_person_system_prompt() -> str:
        return (
            "You are a human survey respondent answering in the first person.\n"
            "You are not an AI system, nor a chatbot.\n\n"
            "Do not use meta-language about being an AI, a model, or a system.\n\n"
            "You are answering as the same speaker across the conversation.\n"
            "Maintain strict self-consistency with what you have already stated about yourself, "
            "your beliefs, and your constraints in prior assistant messages.\n"
            "Do not add new biography, motives, or beliefs unless they have been stated already. "
            "If something is not stated, treat it as unknown.\n\n"
            "Treat prior narrative statements as your own prior statements."
        )

    def prediction_user_prompt(self, outcome: str | None) -> str:
        semantics, schema = self.outcome_semantics(outcome)
        return (
            "Now answer the prediction task.\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def prediction_user_prompt_first_person(self, outcome: str | None) -> str:
        label = self.outcome_label(outcome) if outcome else "the outcome"
        schema = self.outcome_semantics(outcome)[1]
        return (
            "Telling me what you really believe and not what is socially desirable or recommended, "
            "what do you think about whether:\n"
            f"{label}\n"
            "how would you behave:\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def prediction_user_prompt_multi(self, outcomes: list[str]) -> str:
        semantics, schema = self.multi_outcome_semantics(outcomes)
        return (
            "Now answer the prediction task.\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def prediction_user_prompt_first_person_multi(self, outcomes: list[str]) -> str:
        lines = [f"- {o}: {self.outcome_label(o)}" for o in outcomes]
        schema = self.multi_outcome_semantics(outcomes)[1]
        return (
            "Telling me what you really believe and not what is socially desirable or recommended, "
            "what do you think about whether:\n" + "\n".join(lines) + "\n"
            "how would you behave:\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )


__all__ = ["PromptLabelsMixin"]
