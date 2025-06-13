#pragma once
#include <cassert>
#include <string>

#include "linalg/vec.h"
#include "utils.h"

template <typename T>
bool EXPECT_NEAR(T val1, T val2, T tol = 1e-8, T min_tol = 1e-10) {
    if (abs(val1) > min_tol && abs(val2) > min_tol) {
        return abs(val1 - val2) < tol;
    } else {
        return abs(val1) < min_tol && abs(val2) < min_tol;
    }
}

template <typename T>
T rel_err(T val1, T val2, T min_tol = 1e-10) {
    if (abs(val1) > min_tol && abs(val2) > min_tol) {
        return abs(val1 - val2) / abs(val2);
    } else {
        return 0.0;
    }
}

template <typename T>
T abs_err(T val1, T val2) {
    return abs(val1 - val2);
}

// uncolored test report printout
// template <typename T>
// void printTestReport(const std::string& test_name, bool passed, T max_rel_err) {
//     std::string passed_str = passed ? "passed" : "failed";
//     printf("%s %s with rel err %.4e\n", test_name.c_str(), passed_str.c_str(), max_rel_err);
// }

// colored test report printout

template <typename T>
void printTestReport(const std::string& test_name, bool passed, T max_rel_err,
                     bool accept_failure = false) {
    // ANSI escape codes for text color
    const char* color_passed = "\033[32m";   // green
    const char* color_failed = "\033[31m";   // red
    const char* color_error = "\033[34m";    // blue
    const char* color_reset = "\033[0m";     // reset to default
    const char* color_warning = "\033[33m";  // yellow

    const char* result_str = passed ? "passed" : "failed";
    const char* result_color = passed ? color_passed : color_failed;

    // allowable failures but still say it failed
    if (accept_failure && !passed) {
        result_str = "accepted fail";
        result_color = color_warning;
    }

    printf("%s %s%s%s with rel err %s%.4e%s\n", test_name.c_str(), result_color, result_str,
           color_reset, color_error, max_rel_err, color_reset);
}

void printKernelTiming(long long microseconds) {
    // const char* color_time = "\033[33m";  // yellow
    const char* color_time = "\033[2m";   // dim gray/white
    const char* color_reset = "\033[0m";  // reset

    printf("\ttook %s%d%s microseconds to run\n", color_time, (int)microseconds, color_reset);
}

void printKernelTiming(double sec) {
    // const char* color_time = "\033[33m";  // yellow
    const char* color_time = "\033[2m";   // dim gray/white
    const char* color_reset = "\033[0m";  // reset

    printf("\ttook %s%.4e%s sec to run\n", color_time, sec, color_reset);
}

template <typename T>
T rel_err(HostVec<T> vec1, HostVec<T> vec2, T min_tol = 1e-10) {
    assert(vec1.getSize() == vec2.getSize());
    int N = vec1.getSize();
    T max_rel_err = 0.0;
    for (int i = 0; i < N; i++) {
        max_rel_err = std::max(max_rel_err, rel_err(vec1[i], vec2[i], min_tol));
    }
    return max_rel_err;
}

template <typename T>
T abs_err(HostVec<T> vec1, HostVec<T> vec2) {
    assert(vec1.getSize() == vec2.getSize());
    int N = vec1.getSize();
    T max_abs_err = 0.0;
    for (int i = 0; i < N; i++) {
        max_abs_err = std::max(max_abs_err, abs_err(vec1[i], vec2[i]));
    }
    return max_abs_err;
}

template <typename T>
T rel_err(int N, T vec1[], T vec2[], T min_tol = 1e-10) {
    HostVec<T> h_vec1(N, vec1);
    HostVec<T> h_vec2(N, vec2);
    return rel_err(h_vec1, h_vec2, min_tol);
}

template <typename T>
T abs_err(HostVec<T> h_vec1, T vec2[]) {
    HostVec<T> h_vec2(h_vec1.getSize(), vec2);
    return abs_err(h_vec1, h_vec2);
}

template <typename T>
T abs_err(int N, T vec1[], T vec2[]) {
    HostVec<T> h_vec1(N, vec1);
    HostVec<T> h_vec2(N, vec2);
    return abs_err(h_vec1, h_vec2);
}

template <typename T>
T max(int N, T vec[]) {
    T max_val = 0.0;
    for (int i = 0; i < N; i++) {
        max_val = max(max_val, vec[i]);
    }
    return max_val;
}

template <typename T>
T rel_err(HostVec<T> vec1, T vec2[], T min_tol = 1e-10) {
    HostVec<T> h_vec2(vec1.getSize(), vec2);
    return rel_err(vec1, h_vec2, min_tol);
}

template <typename T>
T rel_err(DeviceVec<T> vec1, T vec2[], T min_tol = 1e-10) {
    auto h_vec1 = vec1.createHostVec();
    HostVec<T> h_vec2(vec1.getSize(), vec2);
    return rel_err(h_vec1, h_vec2, min_tol);
}

template <typename T>
T rel_err(DeviceVec<T> vec1, DeviceVec<T> vec2, T min_tol = 1e-10) {
    auto h_vec1 = vec1.createHostVec();
    auto h_vec2 = vec2.createHostVec();
    return rel_err(h_vec1, h_vec2, min_tol);
}

template <typename T>
bool EXPECT_VEC_NEAR(HostVec<T> vec1, HostVec<T> vec2, T tol = 1e-8, T min_tol = 1e-10) {
    assert(vec1.getSize() == vec2.getSize());
    int N = vec1.getSize();
    bool result = true;
    for (int i = 0; i < N; i++) {
        result = result && EXPECT_NEAR(vec1[i], vec2[i], tol, min_tol);
    }
    return result;
}

template <typename T>
bool EXPECT_VEC_NEAR(HostVec<T> vec1, T vec2[], T tol = 1e-8, T min_tol = 1e-10) {
    HostVec<T> vec2_vec(vec1.getSize(), vec2);
    return EXPECT_VEC_NEAR(vec1, vec2_vec, tol, min_tol);
}

template <typename T>
bool EXPECT_VEC_NEAR(DeviceVec<T> vec1, T vec2[], T tol = 1e-8, T min_tol = 1e-10) {
    HostVec<T> h_vec1 = vec1.createHostVec();
    HostVec<T> h_vec2(vec1.getSize(), vec2);
    return EXPECT_VEC_NEAR(h_vec1, h_vec2, tol, min_tol);
}

template <typename T>
bool EXPECT_VEC_NEAR(DeviceVec<T> vec1, DeviceVec<T> vec2, T tol = 1e-8, T min_tol = 1e-10) {
    HostVec<T> h_vec1 = vec1.createHostVec();
    HostVec<T> h_vec2 = vec2.createHostVec();
    return EXPECT_VEC_NEAR(h_vec1, h_vec2, tol, min_tol);
}