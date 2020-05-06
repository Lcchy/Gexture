#include "c74_min.h"
#include <GRT.h>

using namespace GRT;
using namespace c74::min;

class main : public object<main>, public vector_operator<> {
public:
	MIN_DESCRIPTION {"Classify gestures from live sensor feed."};
	MIN_TAGS {"gesture, classification"};
	MIN_AUTHOR {"Lcchy"};

	inlet<>  live_inlet {this, "(signal) Live sensor signal to classify"};
	outlet<> live_output {this, "(signal) Live Id of the recognized gesture", "signal"};


	//Setting up the buffer that will hold the training gestures while recording
	buffer_reference recorded_gestures{ this, MIN_FUNCTION {
								length.touch();
								return {};
							}};

	attribute<number> length {this, "length", 1000.0, title {"Length (ms)"}, description {"Length of the buffer~ in milliseconds."},
		setter { MIN_FUNCTION {
			number new_length = args[0];
			if (new_length <= 0.0)
				new_length = 1.0;

			buffer_lock<false> b {recorded_gestures};
			b.resize(new_length);

			return {new_length};
		}},
		getter { MIN_GETTER_FUNCTION {
			buffer_lock<false> b {recorded_gestures};
			return {b.length_in_seconds() * 1000.0};
		}}};


	//Attributes for toggling options (external API)
	attribute<int> record {this, "record", false, description {"Start recording into given class label."}};
	attribute<bool> train{ this, "train", false, description {"Train the classifier on the dataset."} };
	attribute<bool> classify{ this, "classify", false, description {"Classify the incoming gestures."} };


	//And their respective toggling functions
	message<> record_number_message {this, "number", "Start recording into given class.",
		MIN_FUNCTION {
			record = args[0];
			return {};
		}};
	message<> train_number_message{ this, "number", "Train the classifier on the dataset.",
		MIN_FUNCTION {
			train = bool(args[0]);
			return {};
		} };
	message<> classify_number_message{ this, "number", "Classify the incoming gestures.",
		MIN_FUNCTION {
			classify = bool(args[0]);
			return {};
		} };


	//Create a new DTW instance, using the default parameters
	DTW dtw = DTW(false, true);

	//Load some training data to train the classifier - the DTW uses TimeSeriesClassificationData
	TimeSeriesClassificationData trainingData = TimeSeriesClassificationData(7, "training_gestures");

	//Main signal routine on which Max loops
	void operator()(audio_bundle input, audio_bundle output) {
		auto          in   = input.samples(0);
		auto          out  = output.samples(0);
		buffer_lock<> b(recorded_gestures);


		//Main signal classifying routine (placeholder)
		if (classify) {
			MatrixFloat input_gesture = MatrixFloat();;
			//-->input_gesture.MatrixFloat(in);
			dtw.predict(input_gesture);
			//-->out = Vect(&dtw.getPredictedClassLabel());
		} else {
			for (auto i = 0; i < input.frame_count(); ++i) {
				out[i] = static_cast<float>(in[i]) / 2;
			}
		};
		

		//Recording of one training gesture
		if (record > 0) {
			MatrixFloat sample = MatrixFloat();
			//-->sample.MatrixFloat(in);
			if (!trainingData.addSample(record, sample)) {
				cout << "Failed to record sample!\n" << endl;
			}
		};


		//Train the classifier on the recorded gestures
		if (train) {
			
			//--> Set infotext of trainingData here

			dtw.enableTrimTrainingData(true, 0.1, 90);
			TimeSeriesClassificationData testData = trainingData.split(80);

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

	}

private:
	double m_playback_position {0.0};    // normalized range
	size_t m_record_position {0};        // native range
	double m_one_over_samplerate {1.0};
};


MIN_EXTERNAL(main);
