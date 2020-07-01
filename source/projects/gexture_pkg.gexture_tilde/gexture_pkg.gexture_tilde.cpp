#include <string> 
#include "c74_min.h"
#include <GRT.h>

using namespace GRT;
using namespace c74::min;

#define MAX_WINDOW_SIZE 200 	//Longest gestures length in samples (approx)
#define DATA_DIMENSIONS 1
#define TRIMING_LOW 0.1
#define TRIMING_HIGH 0.9
#define SPLIT_RATIO 0.8

//Naming convention is inconsistent because of conflicts between dependencies conventions
//Sample classes are the labels for training

class gexturer {
public:
	//Declaration of the classifier and his methods
	bool trained;
	MatrixFloat gesture_buffer;
	TimeSeriesClassificationData training_data;
	TimeSeriesClassificationData test_data;
	DTW dtw;

	gexturer() {
		trained = false;
		gesture_buffer = MatrixFloat(MAX_WINDOW_SIZE, DATA_DIMENSIONS);
		training_data.setNumDimensions(DATA_DIMENSIONS);
		test_data.setNumDimensions(DATA_DIMENSIONS);
		DTW dtw = DTW(false, true);
	}

	void push_frames_to_buffer(audio_bundle input) {
	// Pushes the input audio bundle (Max MSP API specific) to the classifier buffer
	// 	Verifiy that input has right number of channels
		if (input.channel_count() != DATA_DIMENSIONS) {
			throw std::runtime_error(
				"Error when loading input, number of channels (" 
				+ std::to_string(input.channel_count()) 
				+ ") must match the data dimension (" + std::to_string(DATA_DIMENSIONS) + ")\n");
		}
		//Extract samples from MAX API audio bundle
		int input_frame_count = input.frame_count();
		MatrixFloat input_samples = MatrixFloat(input_frame_count, DATA_DIMENSIONS);
		GRT::Vector<GRT::Float> temp_col_samples (input_frame_count, 0);
		for (int j=0; j < DATA_DIMENSIONS; j++) {
			for (int i=0; i < input_frame_count; i++) {
				temp_col_samples[i] = float(input.samples(j)[i]);
			}
			if (!input_samples.setColVector(temp_col_samples, j)) {
					throw std::runtime_error("Failed to push input data into temporary buffer at column: " + std::to_string(j) + "\n");
			}
		}
		//Push the values to the classifier buffer
		for (int i = 0; i < MAX_WINDOW_SIZE; ++i) {
			if (i < MAX_WINDOW_SIZE - input_frame_count) {
				//Push the old values back
				if (!gesture_buffer.setRowVector(gesture_buffer.getRowVector(i + input_frame_count), i)) {
					throw std::runtime_error("Failed to push input data into classifier buffer at row: " + std::to_string(i) + "\n");
				}
			}
			else {
				//Push the new values at the end of the buffer
				if (!gesture_buffer.setRowVector(input_samples.getRowVector(i - MAX_WINDOW_SIZE + input_frame_count), i)) {
					throw std::runtime_error("Failed to push input data into classifier buffer at row: " + std::to_string(i) + "\n");
				}
			}
		}
	}

	void add_training_sample(UINT class_label, MatrixFloat sample) {
	//Adds a new sample to the training data under the given label
		if (!training_data.addSample(class_label, sample)) {
			throw std::runtime_error("Failed to save the training sample into class " + std::to_string(class_label) + "!\n");
		}
	}

	void training() {
		if (!dtw.enableTrimTrainingData(true, TRIMING_LOW, TRIMING_HIGH)) {
			throw std::runtime_error("Failed to trim the training data!\n");
		}
		test_data = training_data.split(100 * SPLIT_RATIO);
		if (!dtw.train(training_data)) {
			throw std::runtime_error("Failed to train the classifier!\n");
		}
	}

	float validate() {
	//Use the test dataset to test the DTW model
		double accuracy = 0;
		for (UINT i = 0; i < test_data.getNumSamples(); i++) {
			//Get the i'th test sample - this is a timeseries
			UINT class_label = test_data[i].getClassLabel();
			MatrixDouble timeseries = test_data[i].getData();

			//Perform a prediction using the classifier
			if (!dtw.predict(timeseries)) {
				throw std::runtime_error("Failed to perform prediction for test sample: " + std::to_string(i) + "\n");
			}

			//Get the predicted class label
			UINT predicted_class_label = dtw.getPredictedClassLabel();
			// double maximum_likelihood = dtw.getMaximumLikelihood();
			//Update the accuracy
			if (class_label == predicted_class_label) accuracy++;
			// cout << "TestSample: " << i << "\tClassLabel: " << class_label << "\tPredictedClassLabel: " << predicted_class_label << "\tMaximumLikelihood: " << maximum_likelihood << endl;
		}
		return (accuracy / double(test_data.getNumSamples()) * 100.0);
	}

	int prediction(MatrixFloat sample) {
		if (!dtw.predict(sample)) {
			throw std::runtime_error("Failed to train the classifier!\n");
		}
		int predicted_class_label = dtw.getPredictedClassLabel();
		return predicted_class_label;
	}
};


class main : public object<main>, public vector_operator<> {
public:
	MIN_DESCRIPTION {"Classify gestures from live sensor feed."};
	MIN_TAGS {"gesture, classification"};
	MIN_AUTHOR {"Lcchy"};

	inlet<>  live_inlet {this, "(signal) Live sensor signal to classify"};
	outlet<> live_output {this, "(signal) Live Id of the recognized gesture", "signal"};

	//Attributes for toggling options (external API)
	attribute<int> start_record {this, "start_record", false, description {"Start recording into given class label."}};
	attribute<bool> save_rec {this, "save_rec", false, description {"Save the recorder gestures into the class given at start."}};
	attribute<bool> train{ this, "train", false, description {"Train the classifier on the dataset."} };
	attribute<bool> classify{ this, "classify", false, description {"Classify the incoming gestures."} };

	//And their respective toggling functions
	message<> start_record_number_message {this, "number", "Start recording into given class.",
		MIN_FUNCTION {
			start_record = args[0];
			return {};
		}};
	message<> save_rec_number_message {this, "number", "Save the recorder gestures into the class given at start.",
		MIN_FUNCTION {
			save_rec = bool(args[0]);
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

	gexturer gext = gexturer();

	//Main signal routine on which Max loops through time, feeding signal buffer of inlets as input
	void operator()(audio_bundle input, audio_bundle output) {

		try {
			//Collect input (and output) samples from MAX API
			//and record input samples into the classifier input buffer
			gext.push_frames_to_buffer(input);

			auto out_samples = output.samples(0);

			//Add the current recorded gesture sample to the training data with label start_rec (external API)
			//When done, reinitialize the flags, stop recording and mark classifier as not trained on present data
			if (save_rec) {
				save_rec = false;
				gext.add_training_sample(start_record, gext.gesture_buffer);
				c74::min::object<main>::cout << "Succesfully added the training sample into class " << start_record << endl;
				start_record = 0;
				gext.trained = false;
			};

			// Train and validate the classifier. Result is printed in Max console.
			if (train) {
				train = false;
				gext.training();
				float accuracy = gext.validate();
				c74::min::object<main>::cout << "Test Accuracy: " << accuracy << "%" << endl;
				gext.trained = true;
			};
			
			// Main prediction output
			if (gext.trained) {
				int prediction = gext.prediction(gext.gesture_buffer);
				for (int i = 0; i < input.frame_count(); ++i) {
					out_samples[i] = static_cast<float>(prediction);
				}
			}
			else {
				for (int i = 0; i < input.frame_count(); ++i) {
					out_samples[i] = -1; //External outputs -1 as it is not ready to classify
				}
			}			
		}
		catch (const std::runtime_error& e) {
		//Catching the exceptions thrown by the classifier class here in order to
		//print them to the custom MAX API cerr (Max console)
			c74::min::object<main>::cerr << gext.dtw.getLastErrorMessage() << endl;
			c74::min::object<main>::cerr << gext.training_data.getLastErrorMessage() << endl;
			c74::min::object<main>::cerr << gext.test_data.getLastErrorMessage() << endl;
			c74::min::object<main>::cerr << e.what() << endl;
		}
	}
};

MIN_EXTERNAL(main);
