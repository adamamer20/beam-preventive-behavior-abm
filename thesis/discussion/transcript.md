<!----Introducion & Motivation - 2:20 ---->

The main motivation behind my thesis is the challenge of policymaking. A central problem for policymakers is understanding what would happen if a policy were introduced or changed. In other words, they need to reason about counterfactuals. In the natural sciences, questions of this kind can often be studied through controlled experiments. In the social sciences, however, this is much harder, because policies cannot easily be randomized, populations are heterogeneous, and the underlying dynamics are complex and often poorly understood. One possible response is to move from the experimental domain to th counterfactual reasoning into a computational setting, using agent-based models. Agent-based models are computational models that describe a system from the bottom up. They represent individual actors, the rules that guide their behaviour, and the interactions between them, and then observe the aggregate patterns that result.

But this immediately raises a second question: how should behaviour be modelled realistically inside the ABM? Different ways of modelling individual behaviour can still produce similar system-level outcomes. So even if the model seems to capture the broader pattern, that does not mean the behavioural assumptions inside it are correct. This is also where large language models become interesting. If the challenge is to specify behavioural rules at the micro level, LLMs seem to offer a richer, flexible and realistic way to do that. But that promise needs to be tested.

<!----Proposal 1 min ---->

The goal of my thesis is to build a disciplined link from survey micro-data to both validation of LLMs as behavioural components and simulation.

I begin by building a common empirical backbone of preventive behaviour from survey data.

From there, theI use that backbone as a benchmark to test whether LLMs can reproduce the same behavioural patterns, both at baseline and under controlled changes in individual profiles. Second, I use those empirically estimated behavioural relationships as the core of a survey-grounded ABM to study counterfactual scenarios of preventive behaviour.


<!-----Empirical Backbone Survey data - 1 min ---------->

Let me now move to the first step of the thesis, which is building the empirical backbone.

I start from a large cross-country survey with about 22,000 respondents across six European countries. The survey is very rich: it covers demographics and health, COVID and flu experience, vaccine beliefs, information environment, social exposure, trust, barriers, and moral orientations.

From this survey, I focus on a small set of core prevention outcomes: COVID vaccination willingness, flu vaccination, and non-pharmaceutical preventive behaviours such as masking and staying home when symptomatic.

<!-----Empirical Backbone Reduction pipeline 1:30 min ---------->

The next step is to identify a compact behavioural structure that can explain these outcomes. The raw survey is too high-dimensional and contains too many overlapping items to be used directly for either evaluation or simulation. I therefore reduce it in stages. First, I combine related items into derived variables and indices when they capture the same underlying construct. I then group these into theory-guided behavioural blocks from the prevention literature, such as stakes, trust, legitimacy, disposition, habit, norms, and moral orientation. From these blocks, I estimate reference equations and retain only the components that provide unique empirical signal. The final result is a compact and interpretable empirical backbone, represented as a set of estimated equations that can be carried into both the LLM evaluation and the ABM.


<!-----Empirical Backbone: Prevention compact but outcome specific - 1 min e 45 con la parte del detail sul trust-related blocks ---------->

The first main result is that the same broad blocks recur across outcomes, but their relative importance changes substantially depending on the behaviour.

For COVID vaccination willingness, the dominant block is perceived stakes, with additional roles for institutional trust, legitimacy, and vaccine disposition. <--An interesting detail is that these trust-related blocks are not interchangeable: disposition mostly captures confidence in the vaccine itself, institutional trust captures confidence in the broader public-health institutions, and legitimacy captures acceptance of the vaccine rollout and governance process.-->

For flu vaccination, the dominant driver is habit, which makes this behaviour much more inertial. And when habit is removed, age and health become much more important, so flu vaccination looks less like a routine behaviour and more like a response to perceived medical need.

For the non-pharmaceutical behaviours, the strongest block is moral orientation, especially a more individualizing and prosocial orientation, with stakes again playing a secondary role.

<!--- Second result: which levers matter? 1 min e 45------>

Beyond identifying which blocks carry unique signal, I also study which drivers actually move behaviour when they are shifted in a realistic way.

To do this, I compute reference perturbation effects: the fitted change in the outcome when one variable is moved from a low value to a high value, holding the rest of the profile fixed.

For COVID vaccination willingness, the strongest lever is again stakes, followed by disposition, institutional trust, and legitimacy.

For flu vaccination, short-run leverage is much smaller, which is consistent with the fact that this behaviour is mainly habit-driven.

For the NPI outcomes, the perturbation effects are much more dispersed. This mainly reflects differences in headroom: many respondent profiles are already close to the top or bottom of the predicted range, so the same shift has limited room to change behaviour. The larger effects appear mainly for profiles that start in the middle, where the response can still move.

<!---- LLM Validation ------>

The next step is LLM micro-validation. The question here is whether LLM-based agents can reproduce the empirical benchmark well enough to be credible candidate behavioural components for the ABM.

The benchmark comes from the empirical backbone I built in the previous step. It includes both survey-grounded behavioural outcomes and psychological profile patterns.

I test the models on two targets: behavioural outcomes and a small set of psychological profile variables. And I test them in two settings: first I ask whether the LLM can reconstruct the baseline values from the survey; and second under controlled shifts, where I change one relevant driver and ask whether the model responds in the expected way.

Performance is judged differently in the two settings. At baseline, I look at level accuracy, meaning whether the model predicts the right value, and rank agreement, meaning whether it at least orders respondents correctly from lower to higher propensity. Under controlled shifts, I look at whether the model gets the ordering of perturbation effects right, whether it gets their magnitude roughly right, and whether it avoids reacting when it should not, which is the placebo containment test.

<!---- Baseline result ---->

At baseline, the results are mixed but not trivial.

LLMs can recover part of the cross-sectional structure, especially in ranking respondents from lower to higher propensity. So they are often better at telling who is relatively more or less likely to adopt a behaviour than at predicting the exact point on the response scale.

But level accuracy is weaker, because predictions are compressed toward the middle of the scale. So low values are overpredicted, high values are underpredicted, and heterogeneity is reduced.

This same pattern appears for the psychological-profile side as well. The LLM can recover part of the relative profile structure, but it compresses variation there too. And this matters even more, because compression at the level of the state variables would make simulated agents too similar to one another.

<!---- Perturbed LLM ---->

The more important result comes from the perturbed setting, because this is what matters for counterfactual simulation.

Here performance drops substantially. When I apply controlled shifts in the main behavioural drivers, the models often react too weakly or too strongly, and their responses are too diffuse.

Paired comparisons help somewhat, because they make the contrast between the low and high profile explicit. But the main problem remains: the effect sizes are not well calibrated.

More broadly, the models tend to be too sensitive and too noisy. They often spread the effect too widely across respondents, or across profile dimensions, instead of preserving the more selective structure implied by the empirical benchmark.

So the boundary is quite clear. Off-the-shelf LLMs can recover some baseline structure, but they are not reliable enough as behavioural update mechanisms for a counterfactual ABM.

That is why, in the next step, I return to the survey-grounded relationships and use them as the core of the simulator.

<!--- ABM Specification----->

So, in the final step, I build the agent-based model.

There is however a significant modelling challenge. The survey only gives cross-sectional behavioural relationships, and doesn't give a law of motion it does not directly tell how the same individual updates over time. So the ABM has to bridge that gap in a disciplined way.

The solution I use is a partial-adjustment dynamic. Each agent has a set of mutable behavioural states, and these states move gradually toward time-varying targets. These targets are anchored in the survey-grounded equations, but adjustment is only partial. The key modelling assumption is a local one: near the current state, the between-person gradients estimated from the survey are treated as informative about the direction and relative strength of within-person change. 

Adjustment is allowed to differ across constructs. Trust- and norm-related variables move more slowly, while more situational appraisals, such as perceived danger or infection risk, move faster.

Beyond the dynamic update, the model introduces social interaction explicitly. The contact structure is built from recurring local ties (household-like links), and more occasional contacts. Social influence then works through two main mechanisms. First, peer-pulling. Trust- and risk-related states are pulled toward those of the contacts, with homophily making similar contacts more likely. Second, agents update their descriptive norms for vaccination and NPI behaviour by observing preventive behaviour in their contacts. The rule uses a threshold mechanism: low visible adoption has little effect, but once prevention becomes sufficiently visible, perceived prevalence rises more quickly and then saturates. 

Finally, in addition to social dynamics, The model then closes the loop with a local incidence signal. Each locality carries an incidence signal that can rise, but only within bounds, and that gradually decays unless it is sustained by continued transmission pressure. Higher local protection dampens later incidence, while higher local incidence feeds back into perceived danger and related risk states.

<!--- ABM Results ----->

Once the ABM is specified, I compare a set of counterfactual interventions with different temporal forms, because policy interventions and epidemic shocks do not operate on the same timescale. Some are sustained campaigns, lasting 9 months, because they represent longer-running efforts such as norm or credibility campaigns. Some are one-tick pulses, because they represent short-lived shocks. The outbreak-wave scenario instead represents a temporary but much stronger worsening of epidemic conditions, with a peak around three times the baseline level. Their intensity is standardized using one-sided 1-IQR shifts in the relevant variable, following the same low-to-high convention used earlier in the reference perturbation analysis.

The intervention results reveal four distinct dynamic profiles. Norm-based interventions generate the largest cumulative effects, not because they have the strongest one-step leverage, but because their influence persists and accumulates through social reinforcement and slow adjustment. Outbreak waves, by contrast, produce sharper responses through perceived risk, but these are more short-lived and relax more quickly once epidemic pressure declines. Credibility-related interventions are asymmetric: legitimacy repair mainly increases vaccination willingness, whereas institutional-trust repair has broader spillovers across both vaccination and non-pharmaceutical prevention. If effects are normalized by intervention duration, some credibility shocks, especially legitimacy-related ones, also stand out for their strong short-run impact on vaccination willingness. Access facilitation behaves differently again, because it creates a substitution effect. It can initially raise vaccination, but once protection increases, incidence pressure falls, perceived risk declines, and other precautions may weaken through feedback. So improving one preventive margin does not necessarily raise prevention everywhere else; in the model, this compensatory mechanism is strong enough that the overall net effect turns negative.

<!---- Robutsness screening ------->

Finally, I run a robustness screening using Morris sensitivity analysis.

The idea here is not to ask whether every parameter matters equally, but to ask where uncertainty is concentrated for each intervention–outcome pair.

The resulting pattern is quite interpretable.

Credibility-repair interventions load mainly on persistence. This makes sense, because their effects depend on how slowly trust- and legitimacy-related states relax back toward baseline.

Norm-based interventions depend more on social diffusion. In other words, their yield is strongest when exposure to prevention translates effectively into a shared perception of what is normal.

Outbreak-wave responses are instead concentrated on the incidence loop, which is exactly what we would expect from an intervention that works mainly through changing epidemic pressure and risk salience.

So uncertainty is not spread randomly across the model. It concentrates in a small set of identifiable dynamic assumptions, especially slow-state relaxation, social propagation, and incidence feedback. That makes the ABM easier to interpret, because we can see which mechanisms each policy comparison actually relies on.  

A final useful result is that cumulative comparisons are often more robust than late-horizon comparisons.

This is consistent with the mean-reverting structure of the model. Interventions can separate strongly in total cumulative yield, while ending at much more similar, or even reversed, month-12 levels. So if we care about overall behavioural impact, cumulative measures are often more informative than a single late snapshot. 

<!---- Conclusions ------->

So, stepping back, the main conclusion of the thesis is that preventive behaviour can be represented by a compact but outcome-specific architecture.

The same broad engines recur across outcomes, but their weights change: COVID vaccination willingness depends mainly on stakes within a wider trust and legitimacy context, flu vaccination is much more inertial, and NPIs depend more strongly on moral orientation and stakes.

The LLM results then show that baseline plausibility is not enough for behavioural simulation. Off-the-shelf models can recover part of the static structure, but they are much less reliable once relevant drivers are deliberately perturbed. So the key issue is not only fit at baseline, but counterfactual discipline.

This is why the ABM keeps the behavioural core tied to the survey-grounded relationships and adds a minimal dynamic layer for persistence, social influence, and incidence feedback.

So the broader contribution of the thesis is to propose a disciplined bridge from survey evidence to counterfactual simulation, and to clarify a useful boundary for current LLM agents: they are promising for richer behavioural representation, but not yet reliable enough as unconstrained update engines in this setting.

The main next step would be to strengthen the dynamic side with longitudinal data and a richer transmission layer.