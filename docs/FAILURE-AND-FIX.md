# Failure, diagnosis, and improvement record

## Initial failure

The first training draft used a random train/test split. That made evaluation
look cleaner than the intended deployment scenario because records from later
time periods could influence training while earlier records were evaluated.

## Diagnosis

The problem was not a scikit-learn exception. It was an evaluation design bug:
the split violated the time ordering assumed by the service. A passing test
suite did not prove that the evaluation represented deployment behavior.

## Fix

The training pipeline now sorts by event time and uses the first 70% of records
for training and the final 30% for evaluation. The split function rejects
invalid fractions and has a regression test asserting that every training index
precedes every evaluation index.

## What changed for the better

- Evaluation is now aligned with the serving assumption.
- The model artifact stores feature names and the split strategy.
- The metrics manifest records the evaluation window and sample counts.
- CI runs the regression test on every push.
- Request traces include the model version so future comparisons are possible.

This is a documented engineering improvement, not evidence of real-world model
quality.
