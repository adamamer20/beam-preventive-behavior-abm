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

Finally, I run a robustness screening using Morris sensitivity analysis.

The idea here is not to ask whether every parameter matters equally, but to ask where each intervention’s effect is most sensitive. I group the uncertainty into four broad mechanism families: persistence, social diffusion, incidence feedback, and stochasticity.

The resulting pattern is quite interpretable.

Credibility-focused interventions depend mostly on persistence, which makes sense because their effects are carried by how slowly trust- and legitimacy-related states relax.

Norm campaigns depend most strongly on social diffusion, which is exactly what we would expect if their persistence is sustained by reinforcement and network propagation.

Access facilitation is more sensitive to stochasticity and incidence feedback, because it acts more directly on vaccination and less through the socially reinforcing parts of the model.

A final useful result is that cumulative rankings are often more robust than late-horizon rankings. This is consistent with the mean-reverting structure of the ABM: interventions can differ strongly in total cumulative effect, even if their month-12 levels are much closer together or even reversed.

So the ABM does not just simulate alternative policies. It also helps distinguish between interventions that are sharp but transient, interventions that are gradual but persistent, and interventions whose effects spill across behavioural domains through feedback.