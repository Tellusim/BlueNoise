// MIT License
// 
// Copyright (C) 2018-2023, Tellusim Technologies Inc. https://tellusim.com/
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef __NOISE_BLUE_NOISE_H__
#define __NOISE_BLUE_NOISE_H__

#include <format/TellusimImage.h>
#include <platform/TellusimPlatforms.h>
#include <parallel/TellusimFourierTransform.h>

/*
 */
namespace Tellusim {
	
	/*
	 */
	class BlueNoise {
			
		public:
			
			BlueNoise();
			~BlueNoise();
			
			/// create noise generate
			bool create(const Device &device, uint32_t width, uint32_t height, uint32_t layers);
			
			/// dispatch noise generator
			Image dispatch(const Device &device, const Image &image, uint32_t layers, float32_t sigma, float32_t epsilon);
			
			/// dispatch forward transform
			Image dispatchForward(const Device &device, const Image &image);
			
		private:
			
			/// dispatch generation kernel
			bool dispatch_kernel(const Device &device, Compute &compute, Texture &texture, Kernel &kernel, float32_t value, uint32_t index);
			
			/// print progress
			void print_progress(uint32_t progress, uint64_t begin);
			
			enum {
				MinSize				= 64,
				BatchSize			= 512,
				InverseGroupSize	= 16,
				FilterGroupSize		= 16,
				SampleGroupSize		= 16,
				PositionGroupSize	= 256,
				UpdateGroupSize		= 1,
				RenderGroupSize		= 16,
			};
			
			FourierTransform transform;		// Fourier transform
			
			Kernel inverse_kernel;			// inverse kernel
			Kernel filter_kernel;			// filter kernel
			Kernel min_sample_kernel;		// min sample kernel
			Kernel max_sample_kernel;		// max sample kernel
			Kernel position_kernel;			// position reduction kernel
			Kernel update_kernel;			// update noise kernel
			Kernel render_kernel;			// render noise kernel
			Kernel layer_kernel;			// layer noise kernel
			Kernel upscale_kernel;			// upscale kernel
			
			Texture convolution_texture;	// convolution texture
			Texture forward_textures[2];	// forward textures
			Texture backward_texture;		// backward texture
			Texture upscale_texture;		// upscale texture
			
			Buffer sequence_buffer;			// noise sequence buffer
			Buffer position_buffer;			// noise position buffer
			
			uint64_t old_time = 0;			// old progress time
	};
}

#endif /* __NOISE_BLUE_NOISE_H__ */
