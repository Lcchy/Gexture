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
	string training_file;
	string input_file;
	string output_file;
	DTW dtw;
	TimeSeriesClassificationData trainingData;
	TimeSeriesClassificationData testData;

	//Agent Constructor (interactive instanciation)
	gexturer() {
		cout << "Please enter the Traning file path : ";
		cin >> training_file;
		cout << "Please enter the Input file path : ";
		cin >> input_file;
		cout << "Please enter the Output file path : ";
		cin >> output_file;

		//Create a new DTW instance, using the default parameters
		DTW dtw = DTW(false, true);

		//Load some training data to train the classifier - the DTW uses TimeSeriesClassificationData
		TimeSeriesClassificationData trainingData = TimeSeriesClassificationData(7, "training_gestures");

		dtw.enableTrimTrainingData(true, 0.1, 90);
		TimeSeriesClassificationData testData = trainingData.split(80);
	}

	void train() {
		//Train the classifier on the recorded gestures
		//--> Set infotext of trainingData here

		if (!dtw.train(trainingData)) {
			cout << "Failed to train the classifier!\n" << endl;
		}

		//Use the test dataset to test the DTW model
		double accuracy = 0;
		for (UINT i = 0; i < testData.getNumSamples(); i++) {
			//Get the i'th test sample - this is a timeseries
			UINT classLabel = testData[i].getClassLabel();
			MatrixDouble timeseries = testData[i].getData();

			//Perform a prediction using the classifier
			if (!dtw.predict(timeseries)) {
				cout << "Failed to perform prediction for test sample: " << i << "\n";
			}

			//Get the predicted class label
			UINT predictedClassLabel = dtw.getPredictedClassLabel();
			double maximumLikelihood = dtw.getMaximumLikelihood();
			//Update the accuracy
			if (classLabel == predictedClassLabel) accuracy++;

			cout << "TestSample: " << i << "\tClassLabel: " << classLabel << "\tPredictedClassLabel: " << predictedClassLabel << "\tMaximumLikelihood: " << maximumLikelihood << endl;
		}

		cout << "Test Accuracy: " << accuracy / double(testData.getNumSamples()) * 100.0 << "%" << endl;
	};
};

int main()
{
	gexturer agent = gexturer();

	agent.train();

	//Main signal routine
	for (int i = 0; )
		//Main signal classifying routine (placeholder)
		MatrixFloat input_gesture = MatrixFloat();;
		//-->input_gesture.MatrixFloat(in);
		agent.dtw.predict(input_gesture);
		//-->out = Vect(&dtw.getPredictedClassLabel())
}