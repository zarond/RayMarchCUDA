#include <chrono>


struct TimerInfo {
	std::chrono::high_resolution_clock::rep min;
	std::chrono::high_resolution_clock::rep max;
	std::chrono::high_resolution_clock::rep mean;
	std::chrono::high_resolution_clock::rep dispersion;
};

#pragma once
class CustomTimer
{
private:
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::rep diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//std::chrono::high_resolution_clock::rep mean = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
public:
	int N = 100;
	int index=0;
	std::chrono::high_resolution_clock::rep* times = nullptr;

	CustomTimer() {
		times = new std::chrono::high_resolution_clock::rep[N];
	}
	CustomTimer(int N_): N(N_) {
		times = new std::chrono::high_resolution_clock::rep[N_];
	}
	~CustomTimer() {
		if (times) delete times;
	}

	void tic() { start = std::chrono::high_resolution_clock::now(); }
	void toc(int m=1) { 
		end = std::chrono::high_resolution_clock::now(); 
		diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		times[index % N] = diff/m;
		++index;
	}

	TimerInfo info;
	TimerInfo getInfo() {
		info.min = times[0];
		info.max = times[0];
		std::chrono::high_resolution_clock::rep sum = 0;
		std::chrono::high_resolution_clock::rep sumsqr = 0;
		int H = (index < N) ? index : N;
		for (int i = 0; i < H; ++i) {
			info.min = (times[i] < info.min) ? times[i] : info.min;
			info.max = (times[i] > info.max) ? times[i] : info.max;
			sum += times[i];
			sumsqr += times[i]*times[i];
		}
		info.mean = sum / H;
		info.dispersion = sumsqr/H - info.mean*info.mean;
		return info;
	}
	TimerInfo getInfo(int a) {
		info.min = times[0];
		info.max = times[0];
		std::chrono::high_resolution_clock::rep sum = 0;
		std::chrono::high_resolution_clock::rep sumsqr = 0;
		int H = (index < a) ? index : a;
		for (int i = 0; i < H ; ++i) {
			int j = (index - 1 - i) % N;
			info.min = (times[j] < info.min) ? times[j] : info.min;
			info.max = (times[j] > info.max) ? times[j] : info.max;
			sum += times[j];
			sumsqr += times[j] * times[j];
		}
		info.mean = sum / H;
		info.dispersion = sumsqr / H - info.mean * info.mean;
		return info;
	}

	std::chrono::high_resolution_clock::rep getLastIteration() { return diff; }
};

