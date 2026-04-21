<!----Introducion & Motivation - 2:20 ---->

The main motivation behind my thesis is the challenge of policymaking. A central problem for policymakers is understanding what would happen if a policy were introduced or changed. In other words, they need to reason about counterfactuals. In the natural sciences, questions of this kind can often be studied through controlled experiments. In the social sciences, however, this is much harder, because policies cannot easily be randomized, populations are heterogeneous, and the underlying dynamics are complex and often poorly understood. One possible response is to move from the experimental domain to the counterfactual reasoning into a computational setting, using agent-based models. Agent-based models are computational models that describe a system from the bottom up. They represent individual actors, the rules that guide their behaviour, and the interactions between them, and then observe the aggregate patterns that result.

But this immediately raises a second question: how should behaviour be modelled realistically inside the ABM? Different ways of modelling individual behaviour can still produce similar system-level outcomes. So even if the model seems to reproduce the broader pattern, that does not mean the underlying behavioural mechanism is correct. This is why validating behavioural components becomes central. In recent years, LLMs have attracted attention as candidate behavioural components because they seem to offer a richer, more flexible, and potentially more realistic way to specify behaviour at the micro level. But that promise needs to be tested.

<!----Proposal 1 min ---->

The goal of my thesis is to address these questions by building a disciplined link from survey micro-data to both the validation of LLMs as behavioural components and simulation.

I begin by building a common empirical backbone of preventive behaviour from survey data.

From there, I use those statistical models as the core of a survey-grounded ABM to study counterfactual scenarios of preventive behaviour.

Second, I use that backbone as a benchmark to test whether LLMs can reproduce the same behavioural patterns, both at baseline and under controlled shifts in individual profiles. 

<!-----Empirical Backbone Survey data - 1 min ---------->

The phenomenon I study is preventive behaviour in epidemics.
To build the empirical backbone, I start from a large cross-country survey with about 22,000 respondents across six European countries. The survey is very rich: it covers demographics and health, COVID and flu experience, vaccine beliefs, information environment, social exposure, trust, barriers, and moral orientations.

From this survey, I focus on a small set of core prevention outcomes: COVID vaccination willingness, flu vaccination, and non-pharmaceutical preventive behaviours such as masking and staying home when symptomatic.

<!-----Empirical Backbone Reduction pipeline 1:30 min ---------->

The next step is to identify a behavioural structure that can explain these outcomes. The raw survey is too high-dimensional and contains too many overlapping items to be used directly for either simulation or validation. I therefore reduce it in stages. First, I combine related items into derived variables and indices when they capture the same underlying construct. I then group these into theory-guided behavioural blocks from the prevention literature, such as stakes, trust, legitimacy, disposition, habit, norms, and moral orientation. For each outcome, I retain and identify only the blocks and variables that give unique predictive signals about each outcome. The final result is a set of statistical models for each outcome that can be carried into both the LLM evaluation and the ABM.

<!-----Empirical Backbone: Prevention compact but outcome specific - 1 min e 45 con la parte del detail sul trust-related blocks ---------->

The first main result is that the same broad blocks recur across outcomes, but their relative importance changes substantially depending on the behaviour.

This heatmap should be read column by column. Each column is one prevention outcome, each row is one behavioural block, and a higher score means that block contributes more unique predictive signal for that outcome, once the other blocks are also taken into account.

For COVID vaccination willingness, the dominant block is perceived stakes, with additional roles for institutional trust, legitimacy, and vaccine disposition. 

<!--- An interesting detail is that these trust-related blocks are not interchangeable: disposition mostly captures confidence in the vaccine itself, institutional trust captures confidence in the broader public-health institutions, and legitimacy captures acceptance of the vaccine rollout and governance process. -->

For flu vaccination, the dominant driver is habit, which makes this behaviour much more inertial. And when habit is removed, age and health become much more important, so flu vaccination looks less like a routine behaviour and more like a response to perceived medical need.

For the non-pharmaceutical behaviours, the strongest block is moral orientation, especially a more individualizing and prosocial orientation, with stakes again playing a secondary role.

<!--- Second result: which levers matter? 1 min e 45------>

Beyond identifying which blocks carry unique signal, I also study which drivers actually move behaviour when they are shifted in a realistic way.

To do this, I compute the change in the fitted outcome when one variable is moved from a low value to a high value, holding the rest of the profile fixed.

For COVID vaccination willingness, the strongest lever is again stakes, followed by disposition, institutional trust, and legitimacy.

For flu vaccination, short-run leverage is much smaller, which is consistent with the fact that this behaviour is mainly habit-driven.

For the NPI outcomes, the perturbation effects are much more dispersed. This mainly reflects differences in headroom: many respondent profiles are already close to the top or bottom of the predicted range, so the same shift has limited room to change behaviour. The larger effects appear mainly for profiles that start in the middle, where the response can still move.

<!---- LLM Validation 2 min ------>

The next step is LLM micro-validation. The question here is whether LLM-based agents can reproduce the empirical benchmark well enough to be credible candidate behavioural components for the ABM.

I test the models on two targets: behavioural outcomes, and a small set of psychological profile variables such as trust, legitimacy, perceived stakes, and norms. And I test them in two settings: first at baseline, where I ask whether the model can recover the observed values implied by survey profiles; and second under controlled shifts, where I change one relevant driver and ask whether the model responds in the expected way.

Performance is judged slightly differently depending on the task. At baseline, I ask whether the model gets the level right, whether it ranks respondents correctly from lower to higher propensity, and, for profile variables, whether it recovers the overall profile shape. Under controlled shifts, I ask whether it responds to the right levers, by roughly the right amount, and without generating spurious movement.

<!---- Baseline result 1:20 min ---->

At baseline, the models do recover some structure, but only in a limited sense.

So a higher Gini here means better ordering, while a higher skill score means better level recovery.

For the behavioural outcomes, they are better at sorting respondents relatively than at placing them at the correct point on the response scale. 

The reason is that predictions are systematically pulled toward the centre. Extremes are flattened, and part of the empirical heterogeneity disappear.

A similar pattern appears for the psychological profiles. The model often preserves the broad configuration of the profile, but it shrinks differences across respondents. 

And that is more consequential on the profile side, because these variables are later used to define the agents themselves. If that profile space is compressed, the simulated population starts too homogeneous, tail profiles become too rare, and the model loses part of the heterogeneity that should generate differentiated behavioural responses. 

<!---- Perturbed LLM 1:45 min ---->

The more important test is the perturbed setting, because this is what counterfactual simulation is really about. Policies act by shifting specific behavioural levers, so the key question is whether the model reacts coherently when one of those levers is changed. I test this with controlled perturbations: I shift one driver over a realistic range, keep the rest of the profile fixed, and compare the response with the empirical benchmark.

This figure has two panels. In both panels, moving to the right means better recovery of the empirical gradient ordering: the model becomes better at identifying which profiles should react more. The top panel looks at performance on the active shifts, while the bottom panel looks at containment, meaning whether placebo cells remain quiet instead of moving spuriously.

So, ideally, you would want a method that is far to the right and also high where appropriate, meaning both coherent response and good containment.

The results are clearly weaker than at baseline. The left panel shows a trade-off. In the paired-contrast format, where the model sees the low and high versions of the same profile side by side, perturbation ordering improves: the model becomes better at identifying which profiles should react more. But containment remains weak, so this greater sensitivity also comes with more movement on placebo cells. The right panel shows the second problem: effect sizes are badly calibrated, especially for the NPI outcomes. So the validation step draws a fairly clear boundary. Off-the-shelf LLMs are not yet reliable enough to be used as behavioural components in this setting. So in the final step, when I move to simulation, I build the ABM on the empirical backbone instead.


<!--- ABM Specification 1.50 min ----->

There is however a significant modelling challenge. The survey only gives cross-sectional behavioural relationships, and doesn't give a law of motion, it does not directly tell us how the same individual updates over time. So the ABM has to bridge that gap in a disciplined way.

The solution I use is a partial-adjustment dynamic. Each agent has a set of mutable behavioural states, and these states move gradually toward time-varying targets. These targets are derived from the survey-grounded equations. The key modelling assumption is a local one: near the current state, the between-person gradients estimated from the survey are treated as informative about the direction and relative strength of within-person change. 

Adjustment is also allowed to differ across constructs. Trust- and norm-related variables move more slowly, while more situational appraisals, such as perceived danger or infection risk, move faster.

These targets change over time because the ABM adds two dynamic ingredients. One is social influence. This works in two ways: agents are pulled somewhat toward the people they interact with, and they also update their perceived norms by observing what others do. The second ingredient is incidence feedback: behaviour affects later local incidence, and local incidence in turn feeds back into risk-related targets.

<!---- Dynamic fingerprint 40 secs ------->

Once these relationships are embedded in the ABM, the key point is that interventions do not just differ in size. They also differ in how their effects unfold over time.

This figure should be read row by row. Each row is a different intervention scenario. The first column shows the intervention-related mediator that is being shifted, the second shows the change in vaccination willingness relative to baseline, and the third shows the change in the NPI index relative to baseline.

The norm campaign produces the longest behavioural tail. Its effects remain displaced for longer, consistent with slow adjustment and social reinforcement.

The outbreak wave follows a more classic wave pattern: behaviour rises under higher perceived danger, then relaxes more quickly as pressure fades.

Trust erosion is different again. The initial shock is on credibility, but the more persistent behavioural tail appears mainly in vaccination willingness, while NPIs remain much closer to baseline.

<!---- Conclusions ------->

So, stepping back, I think the thesis leads to three main conclusions.

First, the empirical backbone shows that preventive behaviour is compact, but outcome-specific. The same broad blocks recur across outcomes, but their importance changes depending on the behaviour. COVID vaccination willingness depends mainly on perceived stakes, within a wider trust and legitimacy context. Flu vaccination is much more inertial, and depends more strongly on habit, age, and health. NPIs depend more strongly on moral orientation and stakes.

Second, if we want to use LLMs inside agent-based models, it is not enough to test whether they can recover the baseline reasonably well. They also need to respond coherently under counterfactual changes. And in my tests, that is where current off-the-shelf models still break down.

Third, once these empirical relationships are embedded in an ABM, the policy picture changes. The strongest one-step lever is not always the strongest cumulative lever. Norm-based interventions become especially powerful because their effects persist and reinforce socially. And improving vaccination access alone can also weaken other precautions through feedback, so preventive margins do not always move together.

So the broader contribution of the thesis is to show that it is possible to build a disciplined pipeline from survey micro-data to counterfactual simulation, while also identifying a useful boundary for current LLM agents: they are promising as richer behavioural components, but not yet reliable enough as unconstrained update engines in this setting.

The natural next step is to strengthen the dynamic side with longitudinal evidence on individuals, and to couple the behavioural layer to a richer transmission model.