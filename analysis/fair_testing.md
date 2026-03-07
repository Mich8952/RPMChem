Brainstorming some methods of fair testing of the models.

# Test 1: Strict Test
This is a strict test that compares the raw performance of the vanilla model versus the fine-tuned model
- In this case, neither model should receive additional prompts
- One issue here is that some of the improvement in the results could have come from the ability to learn how to do reasoning (through imputed reasoning)
- A Naive baseline should be considered for the vanilla model - instructing it to reason before providing final solution [test 2].
- Because test 1 doesnt decouple these two aspects
    1. Chemistry fine tuning
    2. Improved reasoning

# Test 2: Prompted Test
- We decouple the issues from test 1 by removing the #2 factor
- This means that we test the same fine-tuned model and compare it to the vanilla model that is prompted to think step-by-step (Chain of Thought)
- We should expect to see the gap decrease from 1 (this whole thing is kind of like an ablation of sorts)

We should repeat the same steps for the numerical tests.