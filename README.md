# Body-Wash-Review-Tagging---Using-GenAI
A Body wash company has multiple reviews but they are not in structured way. Because of this their is no insight.

Attached are two files:
# bodywash-train: 
Mapping of items to Level 1 and Level 2 factors done in this. Think of factors as labels/classification/tags of the item. Level 2 is a more granular association and is nested under Level 1. Each item can be associated with multiple Level 1 and Level 2 factors.
# bodywash-test: 
List of items. As part of the exercise, you have to find the association of these items to Level 1 and Level 2 factors.
Use any LLM model which is suited for NLP task (You can use any free LLM inference service provider e.g. Groq cloud or GCP). Please DO NOT use BERT for this exercise.

# The Approch that i have followed: 

# Approach:
The objective was to automatically assign Level 1 and Level 2 tags to customer reviews. The text data was cleaned and standardised using Python (regex), and predictions were generated using the Gemini 2.5 Flash model through prompt-based classification.

# Model Used:
Gemini 2.5 Flash was selected for its strong contextual understanding, ability to handle zero-shot text classification without additional training, and high speed for processing short textual inputs.

# Output Accuracy:
Since the test data did not have predefined labels, evaluation was done in two ways:
Tag Overlap: Comparing predicted Level 2 tags under common Level 1s with the training distribution showed about 48% overlap.
Manual Validation: A human review of 20 random samples showed contextual accuracy in predictions.
