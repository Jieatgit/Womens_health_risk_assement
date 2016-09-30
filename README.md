# Womens Health Risk Assessment - A MS Azure ML competition project

The objective of this machine learning competition is to build machine learning models to assign a young woman subject (15-30 years old) in one of the 9 underdeveloped regions into a risk segment, and a subgroup within the segment. See more project description  and download data files [HERE](https://gallery.cortanaintelligence.com/Competition/Womens-Health-Risk-Assessment-1). Read detailed data description [HERE](https://az754797.vo.msecnd.net/competition/whra/docs/data-description.docx).

### Summary of work
In this project, I employed different machine learning algorithms, like logistic regression, SVM, random forest and gradient boosting model (GBM). Best performance was achieved by the following models:

* Combined model: sum of predicted probabilities of 11 well chosen models. Accuracy score on test data is 86.70. Ranked 17 on leading board.
* Combined model: selection of predictions according to predicted probability.  Accuracy score on test data is 86.60. Ranked 19 on leading board.
* Random forest with 1000 trees and max depth 13. Accurate score on test data is 86.275438. Ranked 32 on leading board.

Final clean code is in '.py' files. All ipython notebook files were for analysis purpose.  Analysis performed in the project includes:

* Encode categorical features into dummy variables
* Select best a subset of features
* Apply grid search for optimal parameters
* Analyze performance
* Segmentation: partition data and train different models for different segments
* Ensemble models for better performance

More notes:

* RF or gbm models can be easily overfit, that is very small training error. The RF model with 1000 trees and max depth 13 has best validation error, around 0.14, although it has 0.03 train error and it seems over fit.
* Worst prediction happens for data with geo==5 and the accurate score is around 0.6. Partitioning data and training for different models did not work.


It is a great project to practice machine learning algorithms and skills with python sci-kit learn package.
