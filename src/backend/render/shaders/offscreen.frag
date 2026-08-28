#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

uniform float invert;
uniform float color_mode;

// Output gamut correction: linear-light sRGB -> panel-native primaries,
// derived from the output's ICC profile (see utils/icc.rs).
uniform float gamut_enabled;
uniform mat3 gamut_matrix;
uniform vec3 gamut_gamma;

vec3 srgb_eotf(vec3 c) {
    // IEC 61966-2-1 piecewise decode
    vec3 lo = c / 12.92;
    vec3 hi = pow(max((c + 0.055) / 1.055, vec3(0.0)), vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

void main() {
    vec4 color = texture2D(tex, v_coords);

#if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0) * alpha;
#else
    color = color * alpha;
#endif

    // un-multiply
    color.rgb /= color.a;

    // First invert then filter

    if (invert == 1.0) {
        color.rgb = 1.0 - color.rgb;
    }

    if (color_mode == 1.0) {        // greyscale
        float value = (color.r + color.g + color.b) / 3.0;
        color = vec4(value, value, value, color.a);
    } else if (color_mode >= 2.0) {
        float L = (17.8824 * color.r) + (43.5161 * color.g) + (4.11935 * color.b);
	    float M = (3.45565 * color.r) + (27.1554 * color.g) + (3.86714 * color.b);
    	float S = (0.0299566 * color.r) + (0.184309 * color.g) + (1.46709 * color.b);

        float l, m, s;
        if (color_mode == 2.0) { // Protanopia
            l = 0.0 * L + 2.02344 * M + -2.52581 * S;
		    m = 0.0 * L + 1.0 * M + 0.0 * S;
		    s = 0.0 * L + 0.0 * M + 1.0 * S;
        } else if (color_mode == 3.0) { // Deuteranopia
            l = 1.0 * L + 0.0 * M + 0.0 * S;
            m = 0.494207 * L + 0.0 * M + 1.24827 * S;
            s = 0.0 * L + 0.0 * M + 1.0 * S; 
        } else if (color_mode == 4.0) { // Tritanopia
            l = 1.0 * L + 0.0 * M + 0.0 * S;
            m = 0.0 * L + 1.0 * M + 0.0 * S;
            s = -0.395913 * L + 0.801109 * M + 0.0 * S; 
        } else {
            // unknown
            l = L;
            m = M;
            s = S;
        }

        vec3 error;
        error.r = (0.0809444479 * l) + (-0.130504409 * m) + (0.116721066 * s);
        error.g = (-0.0102485335 * l) + (0.0540193266 * m) + (-0.113614708 * s);
        error.b = (-0.000365296938 * l) + (-0.00412161469 * m) + (0.693511405 * s);

        vec3 diff = color.rgb - error;
        vec3 correction;
        correction.r = 0.0;
        correction.g = (diff.r * 0.7) + (diff.g * 1.0);
        correction.b =  (diff.r * 0.7) + (diff.b * 1.0);

        color.rgb += correction;
    }

    // Gamut-compress last: map sRGB-intent colors into the panel's native
    // primaries in linear light, then re-encode with the panel's TRC.
    if (gamut_enabled == 1.0) {
        vec3 linear = srgb_eotf(clamp(color.rgb, vec3(0.0), vec3(1.0)));
        linear = clamp(gamut_matrix * linear, vec3(0.0), vec3(1.0));
        color.rgb = pow(linear, vec3(1.0) / gamut_gamma);
    }

    // re-multiply
    color.rgb *= color.a;

    gl_FragColor = color;
}