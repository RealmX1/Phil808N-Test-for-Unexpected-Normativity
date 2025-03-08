> In case you have read the unpublished research attached to the end of personal statement, this is the same document. 
> Also, the section "analysis of transformer-based LLM's limitation" is the same as the one in my doctoral proposal.
> Code base for this can be found at [https://github.com/RealmX1/Phil808N-Test-for-Unexpected-Normativity](https://github.com/RealmX1/Phil808N-Test-for-Unexpected-Normativity)
# Test for Unexpected Normativity in LLM
> This is ongoing research, and the document is not yet complete.


How does LLM actually store its evaluation mechanism?
> Neural Network is not a computationally efficient way of storing and quering value system.

Index
- [Test for Unexpected Normativity in LLM](#test-for-unexpected-normativity-in-llm)
  - [Introduction](#introduction)
  - [Analysis of transformer-based LLM's Limitation](#analysis-of-transformer-based-llms-limitation)
    - [Intrinsic Limitation of Parametric knowledge storage](#intrinsic-limitation-of-parametric-knowledge-storage)
      - [Recreation of training data and inference of implicit information](#recreation-of-training-data-and-inference-of-implicit-information)
      - [Exhaustive listing of knowledge \& their confidence level](#exhaustive-listing-of-knowledge--their-confidence-level)
      - [Modification of knowledge](#modification-of-knowledge)
      - [Sharing knowledge](#sharing-knowledge)
      - [Conflict in training corpora](#conflict-in-training-corpora)
      - [Redemption: Retrieval Augmented Generation](#redemption-retrieval-augmented-generation)
    - [Limitation of current LLM training paradigm](#limitation-of-current-llm-training-paradigm)
      - [Issue with current LLM training paradigm](#issue-with-current-llm-training-paradigm)
      - [Existing approaches to filter training data](#existing-approaches-to-filter-training-data)
      - [Information Source Grounding](#information-source-grounding)
  - [Unexpected Normativity](#unexpected-normativity)
    - [Two types of normative value enforcement](#two-types-of-normative-value-enforcement)
    - [DPO, RLHF, and Type 2 Normative Value Enforcement](#dpo-rlhf-and-type-2-normative-value-enforcement)
  - [Experiment: Verifying Existence of Unexpected Normativity](#experiment-verifying-existence-of-unexpected-normativity)
    - [Data Generation](#data-generation)
    - [Summarization Generation](#summarization-generation)
    - [Evaluation](#evaluation)
      - [Potential result Analysis](#potential-result-analysis)
    - [Potential Further Analysis](#potential-further-analysis)
  - [Proposal: Separation of Concern between Evaluation and Operation](#proposal-separation-of-concern-between-evaluation-and-operation)
  - [Appendix](#appendix)
    - [Json Schema For Summarization Response Example](#json-schema-for-summarization-response-example)
    - [Additional Question](#additional-question)
  - [Formalization of Problem: Observation based Value System deduction](#formalization-of-problem-observation-based-value-system-deduction)
## Introduction
This "paper" performs an analysis of the limitation of current LLM structure and training paradigm, proposes the hypothesis that these limitation lead to unexpected normativity in LLM's behavior, and conducts an experiment to verify the hypothesis. 

It should be intuitive that when we train LLM with methods like DPO and RLHF, we are training LLM to reject immoral request and align with human normative preference. But existing training methods does not distinguish between task type when performing such normative preference enforcement. This leads to unexpected normativity in some tasks such as text summarization, where additional normative preference is undesirable. (Or, at least, when expressing normative preference, we would want LLM to explicitly state that such preference is not based on the content of the text, but rather on the normative preference of the model itself.) This paper examines this phenomenon, proposes a hypothesis for why this is happening, and conducts an experiment to verify the hypothesis, and eventually proposes a potential solution to the problem.

## Analysis of transformer-based LLM's Limitation
### Intrinsic Limitation of Parametric knowledge storage
Studies on Large Language Model itself have generally focused on the use of models which utilize **parametric knowledge storage** (transformer, mamba, and the more recent study on diffusion based text generation, etc.). This scheme of knowledge storage and retrieval has **major limitations**; It is not able to **consistently** performing the following tasks:
1. **recreate training data** without hallucination or infer implicit information from training data
2. **exhaustively list all knowledge** generalized in a given field (and neither can it do this while consistently providing its confidence level)
3. modify one knowledge instance while
    1. updating related knowledge, and
    2. not affecting other unrelated instance
4. **share knowledge** with other LLM models with different structure, or selectively share knowledge with a different version of the same model
5. **handling conflict in training corpora** - it is especially prominent when using LLM to perform task such as programming, where LLMs are trained on text corpora of code that uses different versions of the same library and as a result sometime produce troublesome output which mix and match usage of different versions

The same limitation also applies to the moral decision making of LLM.

Here I'm going into more detail regarding each of the above issues.
#### Recreation of training data and inference of implicit information
The inability of consistent recreation of training data is rather trivial to prove.
As for the latter, I've performed incomplete testing of the latter - fine tuned LLMs (<=13b) using synthetic data on **statements with different confidence levels** of the claim maker as implicit information, and tested whether the LLM is able to infer the implicit information from the fine-tuning data. Despite LLM being able to identify the relation between the implicit information when provided in context (and provide a reasonable confidence ranking between the symbols that convey the implicit confidence information), and having some capability of reconstructing the statement in training data, **LLMs are not able to answer question regarding the implicit information in it**.  More detail about this can be found in the section after this proposal.

#### Exhaustive listing of knowledge & their confidence level
No experiment or research has been conducted on this specific aspect yet as far as I know, but the concept should be relatively intuitive.
The ability to do so is **essential for the creation of new knowledge**. Without exhaustive listing of knowledge, **repetitive work** is inevitable, and hallucination in knowledge can lead to useless output down the line. Lack of recorded **confidence level** for established knowledge means it would be difficult to perform **ranking among hypotheses** based on different established theories, and what theory to revoke and modify in case of new verified hypotheses come into **conflict** with established ones.

#### Modification of knowledge
Again, I haven't yet found direct research on this specific aspect, but the **[catastrophic forgetting during continuous fine-tuning](https://arxiv.org/abs/2308.08747?utm_source=chatgpt.com)** hints at what can happen with other information when we attempt to modify behavior of LLM through fine-tuning - attempt at modifying one knowledge instance through fine-tuning can lead to loss of information in other knowledge instances.

#### Sharing knowledge
This is rather **self-evident**. Most attempts that try to perform **parameter sharing** between LLM such as layer mashing, despite providing some knowledge sharing, are not able to control **the specific knowledge shared**, and most of the improvement is in terms of **reduction in training time** (serving as an initialization method), rather than improvement in performance. **Mixture of Expert**, first introduced in [2017](https://arxiv.org/abs/1701.06538) and later again in [2021](https://arxiv.org/abs/2112.10684), and now popularized by MistralAI's release of [Mixtral of Experts](https://arxiv.org/pdf/2401.04088) for language model, allow for the utilization of knowledge from different expert models, but it means a drastic **increase in memory usage** (for all of the additional models loaded), and it cannot address the other issues mentioned previously.

#### Conflict in training corpora
The conflict can be two folded - **entity resolution conflict** and **version conflict**.
The survey paper on **[knowledge conflicts for LLMs](https://arxiv.org/pdf/2403.08319)** summarized some prompt engineering research to help LLM resolve entity recognition conflict; but **versioning conflict** doesn't have a well-established solution in inference time. It needs to be addressed from the training data pruning, which can be done as a sub-process of IKGD knowledge instantiation.
The conflict benchmark paper **[ConflictBank](https://arxiv.org/abs/2408.12076)** provided a benchmark for version conflict (what they call temporal conflict), which can be used to test the ability of the new framework proposed in this proposal to handle version conflict.

#### Redemption: Retrieval Augmented Generation
**[Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401)** provided a foundation on which to resolve the above issues. Its main contribution is an approach for **querying large corpora** for relevant information, which, after the dawn of LLM in late 2022 with gpt3, is used to inject into LLM’s context. Later **RAG with knowledge base** brings utilization of **high validity information** with LLM, and in March 2024 Microsoft open sourced the **GraphRAG** framework which has become a widely popular tool for **Graph database based RAG research**.
Until recently, most research regarding RAG focused on using a **static/frozen knowledge base** previously generated by experts. Now, works such as **[iText2KG](https://arxiv.org/abs/2409.03284)** start providing methods for **knowledge structuring from raw text corpus**, which can be used to generate a **dynamic knowledge base** for RAG.

(For more detail about RAG and related topics, see the appendix section.)

But these work still have their limitations, including **not maintaining multiple parallel versions** of subset of knowledge base/graph for sake of exploration, and more importantly, the following issue:


### Limitation of current LLM training paradigm
In short, **grounding LLM in real world observation** is an important step toward **self-improving AI**. This is going to become an increasingly important aspect, as **AI generated content backed by little to no ground truth observation** has started entering training corpora in recent years.

#### Issue with current LLM training paradigm
1. Online Text Corpora before AI is already a **biased sample of observations**, due to multiple factors: the **social-economic status** that affect internet access, the **representation of non-English speaking populations** in mostly English-speaking internet, the propensity of people to talk about **irregular observations** as opposed to the mundane ones, etc. All these factors contribute to:
   The **biased distribution of observation**
2. Most of the claims in the training corpora also suffer from **ungroundedness & unverifiability**, and With growth in prevalence of **AI generated content**, the issue is only worsened.
3. Common logged observation only captures a **relatively limited aspect of a person's observation**. Due to this compression, both in modality and in abstraction, when we try to use such data to verify **theories/hypotheses that require new observation features**, after filtering out the relevant observation, we might **not get enough to form a statistically significant conclusion** -- since we are not able to re-extract relevant features from the original observation data.

#### Existing approaches to filter training data
Many empirical studies such as **Phi-1.5's "Textbooks are all you need"** [[https://arxiv.org/abs/2306.11644]] has shed light on the impact that **"high quality" training data** has on LLM's performance. 
... Need more research on other existing approaches.
This proposal would mainly focus on another approach: directly grounding knowledge in observation data, as it also addresses the third issue mentioned above. It can be used in combination with the previously mentioned approaches.

#### Information Source Grounding
With the recent trend of **Edge AI devices** such as Meta's **Ray-Ban Meta Smart Glasses** and **Limitless pendant**, it is becoming increasingly viable to perform **observation data collection at scale**, which would then inturn be used to augment training data.

Another benefit of having **grounded training data**, is that when a new hypothesis is proposed which requires **additional feature extraction**, the previous data is **not rendered obsolete** as much as it would if only a text corpus is used -- since the information source isn't in compressed text format, **additional labeling** can be done on the raw data. Additionally, **online observation (and possibly, testing)** task assignment to observers can be used to further help with **verification of hypotheses**. 
> Only talking about conducting verifying observations instead of conducting testing here due to limitation of scope but it is definitely desirable to have both; And the IKGD framework has the potential to include interface for integrating active testing.

---

One way in which the above mentioned limitations of LLM manefest themselves in the following way in context of moral decision making is **Unexpected Normativity**:

## Unexpected Normativity
### Two types of normative value enforcement
What do we want when it comes to alignment of LLM behavior?
It should be intuitive that the naive requirement that we would want it to achieve, is to have some sort of "domain of acceptable behavior" (and correspondingly, a domain of unacceptable behavior that it avoids) which is compatible with normative value of the society -- and provide reply accordingly, i.e., denying query in the "unacceptable behavior" domain, and providing reply in the "acceptable behavior" domain. Let's call this "Type 1 Normative Value Enforcement".
But what does LLM actually learn when we perform operations like DPO and RLHF on it? (review 848k for more techniques)
An interesting observation that I've made when using these models, especially when having them summarize text that contain implicit evaluatory information (such as ideological propaganda, news letter with obvious political leaning, my personal philosophical work in value theory, etc.), is that the language models would attempt to perform a "normalization" of the value from the text, in such a way that it align more with the normative value of the society.
What is the model doing in such a situation? Essentially, it is re-evaluating the existing implicit value according to a normative propensity. Let's call this "Type 2 Normative Value Enforcement", or "Unexpected Normativity".
### DPO, RLHF, and Type 2 Normative Value Enforcement
Did LLM learn "acceptable action evaluation"? or did it learn to apply a normativypropensity for all its action? If it learned some of the latter strategy, the reply might contain shift toward some normative value that may not be desirable in application.
It should be intuitive, the second strategy can also help model minimize loss in DPO/RLHF learning, since DPO/RLHF training schema doesn't include any training that discourages such enforcement. 
Here additional testing is needed to verify such hypothesis.


## Experiment: Verifying Existence of Unexpected Normativity
To avoid the inefficacy of individual experiment, ideally a complete experiment should include multiple experiments inspecting the hypothesis from different angles. Here is my currently conducting experiment setup:

### Data Generation
Instead of using real-world texts as done in the preliminary test in proposal for this term paper, to control the variables in the input text we use **synthetically generated data** produced by an advanced LLM (Gemini Flesh 2.0 or ChatGPT-o1). 
A list of event/policy/decision $S=\{s_1, s_2, ..., s_n\}$ the normative opinion is relatively uncontroversial is selected. Each topic is assigned a value indicating deviation from normative leaning, and 20 essays are written, with an aggregated support level value deviating from the normative leaning with the assigned value. Each essay is generated by providing the topic and leaning to the LLM, which would be prompted to generate the essay without using explicit evaluative language or direct/overt signals of the leaning.
The synthetic text is then manually modified to ensure the rule above is enforced.


### Summarization Generation
For each of the testing LLM, for each topic, the LLM is given all synthetic essays on the topic, and asked to perform the following tasks independently (i.e., not within a single conversation):
1. **Summarize**
2. **Summarize, and Identify and Rate Authors' General Leaning**
3. **Summarize, Identify and Rate Authors' Leaning, and Provide (Model's) Personal View**

All responses are in json format and enforced by providing model with respective json schema. See the example in appendix: [Json Schema For Summarization Response Example](#json-schema-for-summarization-response-example)

The reason for testing with these three different prompts is due to the inconsistency between models when it comes to preliminary testing with these different prompts. Some models show better performance when asked to perform the "identify and rate authors' leaning" together with the "summarize" task, while others don't.


### Evaluation
Labeler is shown the evaluation interface, which would include the following information:
1. the topic
2. the synthetic essays and designed aggregate leaning
3. normative leaning
4. response from LLM
And the labeler would be asked to rate how much the llm's summary deviates from the ground truth aggregate leaning, and whether the deviation LLM exhibit is in direction of the normative leaning.

Example for the evaluation interface (I know, it is very ugly and barebone at the moment)
![alt text](image.png)

#### Potential result Analysis
As this experiment in its current test setup would require extensive manual labor (estimated around ~300 hours labeling assuming 1 min per label (an unrealistically fast speed for manual labeling) for 20 topic, 20 essays per topic, 3 models (llama2-7b, llama2-13b, llama2-70b), 3 reply from each model, and 5 repeated label for each response; and even if doing the axis of model-wise comparison and summarization tasks independently, it would still require 2000 labels, or 33 hours labeling), the full experiment isn't conducted yet. The hope is to create a consistent automated labeling process that utilizes advanced LLM and decomposition into atomic tasks, (Or in case any group is interested in funding this in the future, hire human labelers)
Here are some potential results and what can be summarized from them:
1. All LLMs' summary is faithful to the ground truth and does not show statistically significant deviation in direction of normative leaning (or show leaning away from the normative leaning)
Doesn't support the hypothesis.
2. All LLMs' summary shows similar and statistically significant deviation in direction of normative leaning
Strongly support the hypothesis.
3. Some LLM's summary shows deviation towards the normative learning, while some are not
The experiment is inconclusive. If the larger LLM shows less deviation, this might be attributed to smaller LLM being influenced too much by the prompt itself (in which case it shows "unexpected 0-shot learning" instead of "unexpected normativity").

---
Again, no matter what the result is, additional testing using other paradigms is needed to further examine the hypothesis.

---

### Potential Further Analysis
1. **Model Background Leaning Calibration**:
    Different LLM can have different background leaning (background normativity), maybe adjusting the synthetic data generation according to the background leaning of the LLM, and measuring the LLM's deviation from the ground truth toward the background leaning would be a better idea than measuring the LLM's devaition from the ground truth toward the normative leaning. But the issue is that this would mean different LLM would be tested on different synthetic data on the same topic, making comparison between LLM less controlled.
2. **Comparative Analysis**: If multiple articles have different intended leanings (pro, anti, or different subtle biases), compare how well the model detects each type.
3. **Robustness Check**: Vary the complexity of subtleness. Some articles are barely biased; others have very subtle cues. Check if the model’s performance degrades with subtler biases.
4. **Scaling Up**: Use many synthetic articles to produce statistically significant metrics on summary faithfulness, leaning detection, and personal view coherence.


Separation of concern between evaluation and operation can help address this issue.

## Proposal: Separation of Concern between Evaluation and Operation
There are two main approaches to addressing unexpected normativity:
1. Training model's value alignment on a more fine-grained training process. Similar to how [FLASK: FINE-GRAINED LANGUAGE MODEL EVALUATION BASED ON ALIGNMENT SKILL SETS](https://arxiv.org/pdf/2307.10928) does regarding the fine-tuning of instruction following training, a similar schema can be created for the training for value alignment in LLM, by providing different evaluation schema for labelers on different query types, or by introducing an additional axis of evaluation regarding whether the normative value application/enforcement is appropriate.
2. 
(would be part of my future research work in IKGD)


## Appendix
### Json Schema For Summarization Response Example
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ThirdTypeSummarizationResponse",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A summary of the given article, capturing the main points without altering the author's stance."
    },
    "author_leaning": {
      "type": "string",
      "description": "A textual description of the author's subtle leaning or bias inferred from the article."
    },
    "model_personal_view": {
      "type": "string",
      "description": "The model's own stated view or opinion on the topic, provided separately from the author's view."
    }
  },
  "required": ["summary", "author_leaning", "model_personal_view"],
  "additionalProperties": false
}
```

### Additional Question
Before the advent of LLM, most of the conversation we have on a day to day basis with what we perceive as "intelligent being" is in form of discussion another person. Mostly speaking, we never get to speak with a "normative" person all the time, nor do we overwhelmingly speak with a "normative" people -- despite most people in aggregate being "normative", in no sense are they "normative' individually.
But now, increasing percentage of this kind of conversation is conducted with LLM which are trained to be "normative".


---
Unrelated note
## Formalization of Problem: Observation based Value System deduction
This is an extension to what I mentioned after Lixing presented his project.

When considering an arbitrary binary decision $D$, the first person decision maker have a set of observation on factors for and against the decision. Noted as $F_V$ (read as "(decision maker) observed factors for the decision based on moral system $V$"), and use $F_V^{\uparrow}$ and $F_V^{\downarrow}$ to represent the perceived factors in support and against the decision under the moral system $V$.

An evaluation system $V$ is defined as a function mapping from state of decision $S$ and candidate options $O = \{o_1, o_2, \cdots, o_n\}$ to a boolean list $V(S, O) = \{v_1, v_2, \cdots, v_n\}$ of whether each option is permissible.

Each element in $Gnd$ corresponds to a ground truth rollout. As $f$ increases, the chance of the observer to observe the rollout increases, eventually resulting in the observer to observerving the ground truth rollout.

And, for the third party observer, as time goes by, more and more observation about effect of the decision get revealed, noted as $Obs_t$, (read as "observation of effect of the decision till time $t$").
Only part of the rollout is observed by the observer, either due to lack of observation ability, or due to other factors compunding from other decisions affectinghting the observable rollout.


The question is, given observation of a dataset of observed decisions and outcomes of a person's actions $\{(D, F_V^{\uparrow}, Obs_t), \forall D \in D_\text{observed}\}$, how do we approximate $V$ of that person?

The latter also result in observation of what is opposite to the ground truth rollout; and potential false-attribution of the effect on to the decision.


How does it work to assemble into non-binary decision?

> It seems to me that after basic formalization of the problem, that the default logic and deontic logic we learned in seminar doesn't apply very well to this problem. specifically since the value system of individual is prone to change at a shorter notice than the normative value system; and we did not, and I don't think the system in principle is suited for, effective adjustment according to change in value system.
> I also came to the realization that during the last seminar, after Lixing finished presenting his project, my attempt at drawing a parallel between his and this deduction of value system is flawed -- he is suggesting a system for suggesting normative action in case of indecision for people.
