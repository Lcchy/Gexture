#include "c74_min.h"
#include <GRT.h>

using namespace GRT;
using namespace c74::min;

//Naming convention is inconsistent because of conflicts between dependencies conventions
//Sample classes are the labels for training

class main : public object<main>, public vector_operator<> {
public:
	logger max_cerr = cerr;
	MIN_DESCRIPTION {"Classify gestures from live sensor feed."};
	MIN_TAGS {"gesture, classification"};
	MIN_AUTHOR {"Lcchy"};

	inlet<>  live_inlet {this, "(signal) Live sensor signal to classify"};
	outlet<> live_output {this, "(signal) Live Id of the recognized gesture", "signal"};

	//Attributes for toggling options (external API)
	attribute<int> start_record {this, "start_record", false, description {"Start recording into given class label."}};
	attribute<bool> train{ this, "train", false, description {"Train the classifier on the dataset."} };
	attribute<bool> classify{ this, "classify", false, description {"Classify the incoming gestures."} };

	//And their respective toggling functions
	message<> record_number_message {this, "number", "Start recording into given class.",
		MIN_FUNCTION {
			start_record = args[0];
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

	//
	class gexturer {
	public:
		//Declaration of the classifier and his methods
		bool trained;
		int max_window_size; //longest gestures length (approx)
		int *in_buffer[max_window_size];
		int buf_position;
		TimeSeriesClassificationData training_data;
		// --> Possibly set data dims etc here

		gexturer() {

			DTW dtw = DTW(false, true);
		}

		void add_training_sample() {
			if (!training_data.addSample(&start_record, &in_buffer)) {
				cerr << "Failed to save the training sample into class" << &start_record << "!\n" << endl;
			}
		}

		void training() {
			bool bool_return = dtw.train(training_data);
			if (!dtw.enableTrimTrainingData(true, 0.1, 90)) {
				cerr << "Failed to trim the training data!" << endl;
			}
			test_data = training_data.split(80);
		};
	};

	gexturer gext = gexturer();

	//Main signal routine on which Max loops through time, feeding signal buffer of inlets as input
	void operator()(audio_bundle input, audio_bundle output) {
		auto in = input.samples(0);
		auto out = output.samples(0);

		if (start_record > 0) {
			for (int i = 0; i < input.frame_count(); ++i) {
				gext.in_buffer[buf_position + i] = in[i];
			}
			buf_position += input.frame_count();
		};

		//Add the current recorded gesture sample to the training data with label start_rec (external API)
		//When done, reinitialize the flags, stop recording and mark classifier as not trained on present data
		if (save_record) {

			save_rec = false;
			start_rec = 0;
			trained = false;
		};

		// Train and validate the classifier. Result is printed in Max console.
		if (train) {
			training();
			accuracy = gext.validate();
			cout << "Test Accuracy: " << accuracy << "%" << endl
			train = false;
			trained = true;
		};

		
		// Main prediction
		if (trained) {
			prediction = predict(gesture_buffer);
			for (int i = 0; i < input.frame_count(); ++i) {
				out[i] = prediction;
			};
		};

		
		
		
		
		
		
		
		
		
		//Main signal classifying routine (placeholder)
		if (classify) {
			MatrixFloat input_gesture = MatrixFloat();
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

		};

	}
};

MIN_EXTERNAL(main);
