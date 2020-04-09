#include "c74_min.h"
//#include <GRT.h>

//using namespace GRT;
using namespace c74::min;

class main : public object<main>, public vector_operator<> {
public:
	MIN_DESCRIPTION {"Classify gestures from live sensor feed."};
	MIN_TAGS {"gesture, classification"};
	MIN_AUTHOR {"Lcchy"};

	inlet<>  live_inlet {this, "(signal) Live sensor signal to classify"};
	outlet<> live_output {this, "(signal) Live Id of the recognized gesture", "signal"};


	buffer_reference reference_gestures{ this, MIN_FUNCTION {
								length.touch();
								return {};
							}};


	attribute<number> length {this, "length", 1000.0, title {"Length (ms)"}, description {"Length of the buffer~ in milliseconds."},
		setter { MIN_FUNCTION {
			number new_length = args[0];
			if (new_length <= 0.0)
				new_length = 1.0;

			buffer_lock<false> b {reference_gestures};
			b.resize(new_length);

			return {new_length};
		}},
		getter { MIN_GETTER_FUNCTION {
			buffer_lock<false> b {reference_gestures};
			return {b.length_in_seconds() * 1000.0};
		}}};


	attribute<bool> record {this, "record", false, description {"Record into the loop"}};


	message<> number_message {this, "number", "Toggle the recording attribute.",
		MIN_FUNCTION {
			record = args[0];
			return {};
		}};


	message<> dspsetup {this, "dspsetup", MIN_FUNCTION {
						   m_one_over_samplerate = 1.0 / samplerate();
						   return {};
					   }};


	void operator()(audio_bundle input, audio_bundle output) {
		auto          in   = input.samples(0);
		auto          out  = output.samples(0);
		buffer_lock<> b(reference_gestures);

		for (auto i = 0; i < input.frame_count(); ++i) {
			out[i] = static_cast<float>(in[i]) / 2;
		};

		if (b.valid()) {
			auto   position          = m_playback_position;
			auto   frames            = b.frame_count();
			auto   length_in_seconds = b.length_in_seconds();
			auto   frequency         = 1.0 / length_in_seconds;
			auto   stepsize          = frequency * m_one_over_samplerate;

			//for (auto i = 0; i < input.frame_count(); ++i) {
			//	// phasor
			//	position += stepsize;
			//	position = std::fmod(position, 1.0);
			//	sync[i]  = position;

			//	// buffer playback
			//	auto frame = position * frames;
			//	out[i]     = b.lookup(static_cast<size_t>(frame), chan);
			//}
			//m_playback_position = position;

			//	auto record_position = m_record_position;

			//for (auto i = 0; i < input.frame_count(); ++i) {
			//	if (record_position >= frames)
			//		record_position = 0;
			//	b.lookup(record_position, chan) = static_cast<float>(in[i]);
			//	++record_position;
			//}
			//m_record_position = record_position;
			b.dirty();
		}
		else {
			output.clear();
		}
	}

private:
	double m_playback_position {0.0};    // normalized range
	size_t m_record_position {0};        // native range
	double m_one_over_samplerate {1.0};
};


MIN_EXTERNAL(main);
