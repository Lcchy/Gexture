// GextureCLI.cpp : This file contains the 'main' function.
//

#include <iostream>
#include <GRT.h>
using namespace std;
using namespace GRT;


//Main agent class
class gexturer {
public:
	//Declare flags and filenames
	bool debug = true;
	bool stat_validate = false;
	string training_file;
	string input_file;
	string output_file;
	string validation_file;
	DTW dtw;
	TimeSeriesClassificationData training_data;
	TimeSeriesClassificationData test_data;
	MatrixFloat input_data;

	//Agent Constructor (interactive instanciation)
	gexturer() {
		//Get file paths from user
		// cout << "Please enter the Traning file path : ";
		// cin >> training_file;
		// cout << "Please enter the Input file path : ";
		// cin >> input_file;
		// cout << "Please enter the Output file path : ";
		// cin >> output_file;
		// cout << "If necessary, please enter the Validation file path : ";
		// cin >> validation_file;

		if (debug) {
			training_file = "data/prepared_real_data.csv";
			input_file = "data/input_data.csv";
			validation_file = "data/validation_data.csv";
		}

		//Create a new DTW instance, using the default parameters
		dtw = DTW();

		// Smooth a bit the acceleration ! + K fold validation

		dtw.enableScaling(true); //set external ranges ?
		dtw.enableNullRejection(true);
		dtw.setOffsetTimeseriesUsingFirstSample(true);
		dtw.enableZNormalization(true, false);
		if (!dtw.enableTrimTrainingData(true, 0.1, 90)) { // optimize params
			cout << "Failed to trim the training data!\n" << endl;
		}

		dtw.setWarpingRadius(0);
		dtw.setContrainWarpingPath(false);

		load_prepare_data();
	};

	void load_prepare_data() {
		//Load and prepare the training data - the DTW uses TimeSeriesClassificationData
		if (!training_data.loadDatasetFromCSVFile(training_file)) {
			cout << "Failed to load the training data from file!\n" << endl;
		}
		if (!dtw.enableZNormalization(true)) {
			cout << "Failed to enable z-normalization!\n" << endl;
		}
		test_data = training_data.split(80);

		//Load the input data from file
		if (!input_data.loadFromCSVFile(input_file, ', ')) {
			cout << "Failed to load the input data from file!\n" << endl;
		}
	}

	void validate() {
		//Use the test dataset to test the DTW model
		double accuracy = 0;
		for (UINT i = 0; i < test_data.getNumSamples(); i++) {
			//Get the i'th test sample - this is a timeseries
			UINT class_label = test_data[i].getClassLabel();
			MatrixDouble timeseries = test_data[i].getData();

			//Perform a prediction using the classifier
			if (!dtw.predict(timeseries)) {
				cout << "Failed to perform prediction for test sample: " << i << "\n";
			}

			//Get the predicted class label
			UINT predicted_class_label = dtw.getPredictedClassLabel();
			double maximum_likelihood = dtw.getMaximumLikelihood();
			//Update the accuracy
			if (class_label == predicted_class_label) accuracy++;

			cout << "TestSample: " << i << "\tClassLabel: " << class_label << "\tPredictedClassLabel: " << predicted_class_label << "\tMaximumLikelihood: " << maximum_likelihood << endl;
		}

		cout << "Test Accuracy: " << accuracy / double(test_data.getNumSamples()) * 100.0 << "%" << endl;
	};
};

int main()
{
	gexturer agent = gexturer();

	//Train, validate and predict
	if (!agent.dtw.train(agent.training_data)) {
		cout << "Failed to train the classifier!\n" << endl;
	}

	agent.validate();
	
	if (!agent.dtw.predict(agent.input_data)) {
		cout << "Failed to perform prediction for input sample!\n";
	}
	cout << "Predicted Class Label for Input: " << agent.dtw.getPredictedClassLabel() << endl;
	
	//Optional further statistical validation from file
	if (agent.stat_validate) {
		agent.test_data.loadDatasetFromCSVFile(agent.validation_file);
		cout << "----- Statistical validation -----\n" << endl;
		agent.validate();
	}

	return 0;
};