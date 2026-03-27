<!----Introducion & Motivation---->

Policy design is fundamentally a problem of making decisions under uncertainty. What policymakers would ideally like to know is what would happen under alternative interventions before implementing them. In other words, they need counterfactuals.

In the natural sciences, counterfactuals are often studied through controlled experiments. In the social sciences, this is much harder: many policies cannot be easily randomized, and outcomes are shaped by heterogeneous individuals, mutual influence, and feedback over time.

For this reason, one way to study counterfactuals is to move this experimental logic into computation, and agent-based models are a natural tool for doing so. In an agent-based model, we simulate a system from the bottom up, starting from individual agents, their behaviour, and their interactions. But ABMs also face a major problem: different micro-level mechanisms can generate similar macro-level outcomes, so aggregate fit alone is not enough.

This is also where large language models become interesting. If the challenge is to specify behavioural rules at the micro level, LLMs seem to offer a richer and more flexible way to do that. But that promise needs to be tested.

<!----Proposal---->

So the goal of my thesis is to build a disciplined pipeline from survey micro-data to agent-based simulation.

I do this in three steps.

First, I build a common empirical backbone of preventive behaviour from survey data.

Second, I test whether LLMs can reproduce these empirical patterns, both at baseline and under controlled shifts in profile or context.

Third, I use these behavioural relationships as the core of an ABM and study counterfactual scenarios of preventive behaviour.

<!-----Empirical Backbone---------->

Let me now move to the first step of the pipeline, which is building the empirical backbone.

I start from a large cross-country survey with about 22,000 respondents across six European countries. The survey is very rich: it covers demographics and health, COVID and flu experience, vaccine beliefs, information environment, social exposure, trust, barriers, and moral orientations.

From this survey, I focus on a small set of core prevention outcomes: COVID vaccination willingness, flu vaccination, and non-pharmaceutical preventive behaviours such as masking and staying home when symptomatic.

The challenge is that the questionnaire is too high-dimensional to use directly. So I reduce it to a compact empirical backbone. I construct derived variables and indices, group them into theory-guided blocks from the prevention literature, estimate survey-grounded equations, and retain only the components that provide unique signal once the others are taken into account.

The point of this reduction is to obtain a compact and interpretable representation that can later be carried into both the LLM evaluation and the ABM.

The first main result is that the same broad blocks recur across outcomes, but their relative importance changes substantially depending on the behaviour.

For COVID vaccination willingness, the dominant block is perceived stakes, with additional roles for institutional trust, legitimacy, and vaccine disposition. An interesting detail is that these trust-related blocks are not interchangeable: disposition mostly captures confidence in the vaccine itself, institutional trust captures confidence in the broader public-health institutions, and legitimacy captures acceptance of the vaccine rollout and governance process.

For flu vaccination, the dominant driver is habit, which makes this behaviour much more inertial. And when habit is removed, age and health become much more important, so flu vaccination looks less like a routine behaviour and more like a response to perceived medical need.

For the non-pharmaceutical behaviours, the strongest block is moral orientation, especially a more individualizing and prosocial orientation, with stakes again playing a secondary role.

<!--- Second result: which levers matter?------>

For policy and simulation, it is not enough to know which drivers are important in the fitted equations. We also want to know which variables actually move behaviour when they are shifted in a realistic way.


So I compute what I call a reference perturbation effect. I move one retained variable from a typical low value to a typical high value, while keeping the rest of the profile fixed, and I measure the implied change in the fitted outcome.

The results are broadly consistent with the previous picture.

For COVID vaccination willingness, the strongest lever is again stakes, followed by disposition, institutional trust, and legitimacy.

For flu vaccination, short-run leverage is much smaller, which is consistent with the fact that this behaviour is mainly habit-driven.

For the NPI outcomes, the perturbation effects are much more dispersed. This mainly reflects differences in headroom: many respondent profiles are already close to the top or bottom of the predicted range, so the same shift has limited room to change behaviour. The larger effects appear mainly for profiles that start in the middle, where the response can still move.

<!---- LLM Validation ------>

The next step is the LLM validation. The question here is whether LLM-based agents can reproduce the survey-grounded empirical benchmark in a way that is actually useful for simulation.

I evaluate the models in two settings. In the unperturbed setting, I ask whether the LLM can reconstruct the baseline pattern in the survey. In the perturbed setting, I shift one relevant driver and ask whether the model reacts in the right way.

I evaluate both level accuracy and rank agreement. In other words, I ask not only whether the LLM predicts the right value, but also whether it at least orders respondents correctly from lower to higher propensity.

I run the same logic on behavioural outcomes and also on a small set of psychological profile variables, because these are the state dimensions that would later carry heterogeneity into the simulator.

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

<!--- ABM ----->
