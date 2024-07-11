using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommender.Console;


var mlContext = new MLContext();

(IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

EvaluateModel(mlContext, testDataView, model);

(IDataView training, IDataView test) LoadData(MLContext mlContext)
{
    var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "data", "recommendation-ratings-train.csv");
    var testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "recommendation-ratings-test.csv");

    var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
    var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

    return (trainingDataView, testDataView);
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
{
    //Define the data transformations
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

    //Choose the machine learning algorithm and append it to the data transformation definitions
    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "movieIdEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 20,
        ApproximationRank = 100
    };

    //We are using MatrixFactorizationTrainer as it is a common approach for recommendation when data exists on how 
    //users have rated products in the past
    //See here for othere recommended algorithms: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/movie-recommendation#other-recommendation-algorithms
    var trainingEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");

    //Fit the model to the Train data and return trained model

    //The Fit() method trains your model with the provided training dataset.
    //Technically, it executes the Estimator definitions by transforming the
    //data and applying the training, and it returns back the trained model, which is a Transformer.
    ITransformer model = trainingEstimator.Fit(trainingDataView);

    return model;
}

void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
{
    //Once you have trained your model, use your test data to evaluate how your model is performing.

    Console.WriteLine("=============== Evaluating the model ===============");

    //The Transform() method makes predictions for multiple provided input rows of a test dataset.
    var prediction = model.Transform(testDataView);

    //Evaluate the model.

    //Once you have the prediction set, the Evaluate() method assesses the model,
    //which compares the predicted values with the actual Labels in the test dataset and returns metrics on how the model is performing.
    var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

    //The root of mean squared error (RMS or RMSE) is used to measure the differences between the model predicted values
    //and the test dataset observed values. Technically it's the square root of the average
    //of the squares of the errors. The lower it is, the better the model is.

    //R Squared indicates how well data fits a model. Ranges from 0 to 1. A value of 0 means
    //that the data is random or otherwise can't be fit to the model.
    //A value of 1 means that the model exactly matches the data. You want your R Squared score to be as close to 1 as possible.

    Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
    Console.WriteLine($"RSquared: {metrics.RSquared}");
}