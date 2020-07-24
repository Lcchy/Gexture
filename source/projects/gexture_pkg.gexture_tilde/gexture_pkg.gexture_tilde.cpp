#include <string> 
#include "c74_min.h"
#include <GRT.h>
#include <chrono>


using namespace GRT;
using namespace c74::min;

#define SENSOR_DATA_RATE 100  	// Incoming samplerate of the sensor data, independent of the Max MSP samplerate
#define MAX_WINDOW_SIZE 200 	// Longest gestures length in samples (approx)
#define DATA_DIMENSIONS 1
#define TRIMING_LOW 0.1
#define TRIMING_HIGH 90
#define SPLIT_RATIO 80

//Naming convention is inconsistent because of conflicts between dependencies conventions
//Sample classes are the data labels for training

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
		
		//Setting the appropriate statistical pre processing here
		dtw.enableScaling(true); //set external ranges ?
		dtw.enableNullRejection(true);
		dtw.setOffsetTimeseriesUsingFirstSample(true);
		dtw.enableZNormalization(true, false);
		if (!dtw.enableTrimTrainingData(true, 0.1, 90)) { // optimize params eventually
			throw std::runtime_error("Failed to trim the training data!\n");
		}

		dtw.setWarpingRadius(0);
		dtw.setContrainWarpingPath(false);
	}

	void push_frames_to_buffer(audio_bundle input, double max_msp_signal_samplerate) {
	// Pushes the input audio bundle (Max MSP API specific) to the classifier buffer
		// 	Verifiy that input has right number of channels
		if (input.channel_count() != DATA_DIMENSIONS) {
			throw std::runtime_error(
				"Error when loading input, number of channels ("
				+ std::to_string(input.channel_count())
				+ ") must match the data dimension (" + std::to_string(DATA_DIMENSIONS) + ")\n");
		}
		//Extract samples from MAX API audio bundle into temp MatrixFloat "input_samples"
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
		test_data = training_data.split(SPLIT_RATIO);
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

	//Attributes for toggling options (Max external API)
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

		double max_msp_signal_samplerate = samplerate();

		try {
			//Collect input buffers (and output, both containing 64 samples by default) from MAX API
			//and record input samples into the classifier input buffer
			gext.push_frames_to_buffer(input, max_msp_signal_samplerate);

			auto out_samples = output.samples(0);

			//Add the current recorded gesture sample to the training data with label start_rec (external API)
			//When done, reinitialize the flags, stop recording and mark classifier as not trained on present data
			if (save_rec) {
				save_rec = false;
				cout << max_msp_signal_samplerate << endl;
				cout << std::to_string(input.frame_count()) << endl;
				cout << tdiff_m << endl;
				cout << tdiff_n << endl;
				std::chrono::steady_clock::time_point tot_end = std::chrono::steady_clock::now();
				cout << std::chrono::duration_cast<std::chrono::nanoseconds> (tot_end - tot_begin).count() / counter << endl;
				
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
					
		//Set the vector size of the incoming audio chunk (audio bundle) to be
		//large enough to reduce the iteration frequency of operator to match the SENSOR_DATA_RATE
		//That way, the model has enough time to predict and enough data is incoming to extract one sample.
			// Main prediction output
			if (gext.trained) {
				int prediction = gext.prediction(gext.gesture_buffer);
				cout << prediction << endl;
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
			if (gext.dtw.getLastErrorMessage() != "") {
				c74::min::object<main>::cerr << gext.dtw.getLastErrorMessage() << endl;
			}
			if (gext.training_data.getLastErrorMessage() != "") {
				c74::min::object<main>::cerr << gext.training_data.getLastErrorMessage() << endl;
			}
			if (gext.test_data.getLastErrorMessage() != "") {
				c74::min::object<main>::cerr << gext.test_data.getLastErrorMessage() << endl;
			}
			c74::min::object<main>::cerr << e.what() << endl;
		}
	}
};

MIN_EXTERNAL(main);
