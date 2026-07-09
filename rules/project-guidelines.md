D&G Project

Deep Learning & Generative AI Project        
(Diploma Level of BS in Data Science and Applications)
                                              (Latest evaluation policy: Click here)

Overview
This project is designed to give students hands-on experience in end-to-end machine learning workflows, from dataset handling and model training to evaluation, deployment, and reporting.

You will:

Work with GitHub for code and project organization
Use Kaggle for competition-style inference and benchmarking
Track experiments and metrics with Weights & Biases (W&B)
(Optional Bonus) Deploy your final model on Hugging Face Spaces
The duration for this project will be around  8 weeks.


Kaggle Competition Link: <<Click here>>

Project registration (Setup Only)
This is an individual project; please create accounts on the following platforms (if not already done):

Kaggle
GitHub
Weights & Biases
Hugging Face
What to do now (for registration)
Step-by-Step: GitHub Repo Setup + Credential Validation for DL & GenAI Course Project
        Slides link: DL GenAI Project  - registration process(Jan-26)

Kaggle Competition Setup
Join the Kaggle competition using this link: <<click here>>
After joining, go to the Code tab and create a new notebook.
Name your notebook as:
DL-YourRollNo-notebook-t22026
 Example: DL-21f1001234-notebook-t22026
Share the notebook with the competition admin: dlgenaiproject (View access is enough, keep it private).
Make at least one baseline submission (even a dummy model is fine).
Note: Later, you may complete the training separately in Kaggle or Colab, then upload the trained models to

Kaggle Hub and import them into your inference notebooks while submitting.

GitHub repo setup
Create a repository for your project (private with instructor access).
Add a basic README.md containing:
Project title (tentative)
Your name & ID
Empty folder structure (e.g., /scripts, /notebooks, /data).
(You will add actual code and notebooks later during the project.)

        Note: Sample GitHub repo structure & guideline for reference

Weights & Biases (W&B) setup
Create a new W&B project with the name:
YourRollNo-t22026
Final Step:
Once you have completed the above setup, submit the [Registration Form] with your Kaggle username, GitHub repo link, and W&B project URL.

(You will add scripts, notebooks, and reports later during the project milestones.)

Note: Without completing all three setups (Kaggle, GitHub, W&B), submitting at least 2 milestones, and submitting Form 1, you will not be eligible for evaluation or viva.

4. Duration of the project
Release of project statement:                     May 22, 2026
Latest Viva & Evaluation Policy:                  click here
Project Evaluation Timelines:

DG Course theory completion Term

Deadline to Cross Cutoff

Deadline to Complete Both Vivas


Sept 2025


30 June 2026

12 July 2026


15 July 2026

30 July 2026


Jan & May 2026


15  July 2026

30 July 2026


31 July 2026

15 Aug 2026


5. Grading Formula
The formula for computing Total Score (out of 100) is as follows:


Final marks =  KP + C+ R + V +M


Component

Weightage

Evaluated By

Details / Sub-criteria

Kaggle Performance (KP)

30 marks

Auto (Leaderboard)

Based on the student’s relative best performance on the Kaggle leaderboard.



T2-2026 Cutoff Score -  <<0.73>>

Notebook & Code Quality(C)

5 marks

L1 Examiner

Quality of code in GitHub repo/Notebook. Checked for readability, modularity, proper documentation, and reproducibility (requirements.txt / environment.yaml).

Report (R)

15 marks

L2 Examiner

Concise technical report (max 5-6 pages) covering problem statement, methodology, model architecture, training details, evaluation, error analysis, and insights.

   Sample format: Click here

Viva (V)

40 marks

L2 Examiner

Individual understanding tested through viva: model design, concepts in DL/GenAI, justification of choices, and ability to extend/critique the solution.

Milestones (M)

10 marks

Auto & TA verify

Marks for meeting deadlines:

Completing the registration process successfully.
Submitting initial training and fine-tuning code on GitHub.
Submit improved inference notebook(s) to Kaggle.
Submit report
Final submission.
     Bonus: (Deployment)

+5 marks

Auto & TA verify

Additional marks for deploying the model/app (e.g., Hugging Face Space, Streamlit, or Flask API). Evaluated on usability and stability.

5a. Criteria to pass this project course
Students must meet the following requirements:

Model Requirements

Present at least three unique models:
One model built from scratch
One pretrained model
One additional model of choice
Experiment Tracking

All models must have valid Weights & Biases (WandB) runs
At least three runs must be compared using common evaluation metrics such as accuracy and F1 score
Development Practice

GitHub commit history must reflect consistent progress over more than three weeks
Last-minute uploads without meaningful commit history will not be accepted
Viva Expectations

Clearly explain the code, demonstrating a strong understanding of its functionality and purpose
Write a small function or code snippet during the viva to validate hands-on work
Milestone Requirement

Submission of at least 2 out of 5 milestones is mandatory to be eligible for the viva
Kaggle Requirement

A minimum score cutoff of <<0.73>> is required in the final Kaggle submission to qualify for the viva
Passing Criteria

L2 Viva score must be at least 20 out of 40
Total project score must be at least 50 out of 100
Projects that do not meet the above requirements will not be considered for evaluation.

6. Forms to be Filled:

Form No.

Form Name

Deadline

Submission Link / Details

Form 1

Registration Form

30/06/2026

  <<click here>>

Form 2

Report Submission Form

31/07/2026

 <<click here>>

Form 3

Deployment Link

31/07/2026

 <<click here>>



Tentative Viva Schedule
July 1st–31st July, 2026 (slots will be announced later)

7. Important Links:

Orientation session recording:     Click here
Registration  & L1 eligibility Status:  <<Yet to be added>>
 Viva timeline:                         <<Click here>>

8. Milestone Plan

Milestones

Deadline

Description

Milestone 0

Jun 22, 2026


Thursday


Orientation & Setup

Attend the orientation session.
Create accounts on Kaggle, GitHub, and Weights & Biases (W&B).
Verify environment setup and tools access.
Milestone 1

Jun 24, 2026


Wednesday


NLP Foundation & Semantic Similarity

Perform Text cleaning, tokenization, and handling missing data.
Generate embeddings using baseline models like TF-IDF and Word2Vec.
Compute cosine similarity between a 'prompt' and 'options' and understand its concepts.
Calculate mAP@3 and understand the concept of mAP.
Milestone 2

Jul 1, 2026


Wednesday


Enter the Transformers

Introduction to the Hugging Face transformers and datasets libraries.
Learn architecture of BERT/RoBERTa and the concept of attention mechanisms.
Use pre-trained embedding models to generate context-aware embeddings.
Zero-shot classification concepts with transformers and SLMs
Milestone 3

Jul 8, 2026


Wednesday


Context Augmentation with RAG Pipelines

The limitations of General LLMs. Learn the RAG pipeline in brief.
Loading a simple pre-built vector database.
Retrieving external context based on the question prompt.
Feeding the retrieved context + prompt + choices into the model to improve reasoning.
Milestone 4

Jul 15, 2026


Wednesday


Formulating MCQ Task & Fine-Tuning

Data formatting for MCQ: Concatenating the question with options.
Introduction to LoRA-finetuning, and its advantages over Full-finetuning.
Setting up a training loop to fine-tune the model weights on the dataset.
Managing GPU memory and batch sizes.
Training efficiency strategies with Training arguments.
Milestone 5

Jul 21, 2026


Wednesday

Ensembling

Extracting and sorting logits to get the top 3 predictions (e.g., A C B).
Ensemble techniques to stack predictions from multiple models.
Other strategies to improve predictions.
Last Submission

26-07-2026


Sunday

Final Submission & Presentation

Make final Kaggle submission.
Create report & Present results and analysis (Macro F1, Error Analysis, Insights).
Optional: Deploy model using Streamlit/Gradio.

9. L1 viva Checkpoints:

The L1 Examiners will verify the following items during Level 1.

1. Model Completion
 The student must present at least three unique models. One must be a model built from scratch. One must be a pretrained model. The third can be any model of their choice.

2. Wandb Tracking
 All models must have valid Wandb runs. At least three runs must be compared using common metrics such as F1 score and accuracy.

3. GitHub or Notebook Authenticity
 Commit history must show steady progress spread across more than three weeks. Last-minute uploads without meaningful history will not be accepted.

4. Live Coding Check
 The student should be able to write a small function or code block during the viva to confirm hands-on work.

5. Code Walkthrough
 Student must clearly explain their code flow. This includes data processing, model setup, training logic, and evaluation steps.

6. Report Understanding
 Student must show that they understand their own report. They should be able to explain their method, observations, and results.

7. Authenticity Check
 The examiner will confirm that the project is the student’s own work and not prepared by someone else. He/She may ask questions about your report/code, etc.

Note: All 7 points are mandatory to pass the L1 viva.


10 L2-Viva Overview

Level 2 Viva will be taken by industry professionals. They will check the student’s depth of understanding, the quality of decisions made during the project, and how confidently the student can explain and defend their work. This round focuses on technical clarity, reasoning, and end-to-end thinking.        


11 FAQs:

Q1: Do I need to take DL/GenAI theory in the same term?
Ans: No, you may complete the course first, then the project later. (It is highly recommended to take up the project after completing the DL & GenAI course.)

Q2: Can I take the viva before the deadline?
Ans: Yes, but the Kaggle score component will be computed after the deadline.

Q3: What if I can’t improve my score?
Ans: Focus on a strong report, multiple models, and a clean repo – marks come from all components.

Q4: What if the Hugging Face deployment fails?
Ans: You can still pass, but you’ll lose 5 marks.

Q5: What will be checked in L1 viva
 The Level 1 viva checks the authenticity of your work and readiness for Level 2. Key points include:

Code Structure & Modularity: Preprocessing, training, and inference should be in separate scripts.
Comments & Documentation: Code should have clear explanations and comments.
GitHub Repository: Repo should have proper structure, README, setup instructions, and dependencies.
Model Completion: At least three models – one from scratch, one pretrained, and a third of your choice.
Wandb Tracking: Compare at least three model runs using metrics like F1 score and accuracy.
Code Understanding & Live Coding: Be able to explain your code and write a small piece during the viva.
Report & Future Work: Understand your report and clearly state the next steps.

Q6. What should I include in my project report?

Ans: Your report should clearly explain the models you used, including diagrams and architecture details. Describe your training process, dataset splits, any data augmentation, and the overall pipeline. Include hyperparameter tuning details with logs or trial results. Present evaluation metrics relevant to your task, with tables or plots. Finally, provide error analysis with examples of model failures and insights drawn from them.