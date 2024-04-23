#pragma once

#ifndef FILE_RANDOMSAMPLERH_SEEN
#define FILE_RANDOMSAMPLERH_SEEN

class RandomSampler {
public:
	RandomSampler();
	RandomSampler(unsigned int id);
	RandomSampler(unsigned int pixelId, unsigned int sampleId);
	RandomSampler(unsigned int x, unsigned int y, int sampleId);


	/* Seperate constructor from setting the seed with same inputs */
	inline void seed(unsigned int id);
	inline void seed(unsigned int pixelId, unsigned int sampleId);
	inline void seed(unsigned int x, unsigned int y, int sampleId);

	float get_float();
	int get_int();

private:
	unsigned int MurmurHash3_mix(unsigned int has, unsigned int k);
	unsigned int MurmurHash3_finalize(unsigned int hash);
	unsigned int LCG_next(unsigned int value);

	unsigned int m_s;

};

RandomSampler::RandomSampler() {
	/* Use a seed function rather than this default */
	seed(0);
}

RandomSampler::RandomSampler(unsigned int id) {
	seed(id);
}



RandomSampler::RandomSampler(unsigned int pixelId, unsigned int sampleId) {
	seed(pixelId, sampleId);

}

RandomSampler::RandomSampler(unsigned int x, unsigned int y, int sampleId) {
	seed(x | (y << 16), sampleId);

}

void RandomSampler::seed(unsigned int id) {
	unsigned int hash = 0;
	hash = MurmurHash3_mix(hash, id);
	hash = MurmurHash3_finalize(hash);

	m_s = hash;
}

void RandomSampler::seed(unsigned int pixelId, unsigned int sampleId) {
	unsigned int hash = 0;
	hash = MurmurHash3_mix(hash, pixelId);
	hash = MurmurHash3_mix(hash, sampleId);
	hash = MurmurHash3_finalize(hash);

	m_s = hash;

}

void RandomSampler::seed(unsigned int x, unsigned int y, int sampleId) {
	seed(x | (y << 16), sampleId);
}


unsigned int RandomSampler::LCG_next(unsigned int value)
{
	const unsigned int m = 1664525;
	const unsigned int n = 1013904223;

	return value * m + n;
}

int RandomSampler::get_int() {
	m_s = LCG_next(m_s); return m_s >> 1;
}

float RandomSampler::get_float() {
	return (float)get_int() * 4.656612873077392578125e-10f;

}

unsigned int RandomSampler::MurmurHash3_mix(unsigned int hash, unsigned int k) {
	const unsigned int c1 = 0xcc9e2d51;
	const unsigned int c2 = 0x1b873593;
	const unsigned int r1 = 15;
	const unsigned int r2 = 13;
	const unsigned int m = 5;
	const unsigned int n = 0xe6546b64;

	k *= c1;
	k = (k << r1) | (k >> (32 - r1));
	k *= c2;

	hash ^= k;
	hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;

	return hash;
}
unsigned int RandomSampler::MurmurHash3_finalize(unsigned int hash) {
	hash ^= hash >> 16;
	hash *= 0x85ebca6b;
	hash ^= hash >> 13;
	hash *= 0xc2b2ae35;
	hash ^= hash >> 16;

	return hash;
}



#endif //FILE_RANDOMSAMPLERH_SEEN