// magnus_mkl_complex.cpp
// Пример: метод Магнуса (midpoint) + expm (scaling&squaring + Pade13) для комплексных матриц.
// Использует Intel MKL (cblas_zgemm, LAPACKE_zgetrf, LAPACKE_zgetrs и т.д.)

#include <mkl.h>
//#include <lapacke.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
using namespace std;

using complexd = std::complex<double>;
using CMatrix = std::vector<complexd>; // row-major N*N

// ------ utility: index for row-major ------
inline int idx(int i, int j, int N) { return i * N + j; }

// ------ simple helpers ------
CMatrix eye(int N) {
    CMatrix I((size_t)N * N, complexd(0.0, 0.0));
    for (int i = 0; i < N; ++i) I[idx(i, i, N)] = complexd(1.0, 0.0);
    return I;
}

void printMat(const CMatrix& A, int N, const string& name = "A") {
    cout << name << " (" << N << "x" << N << "):\n";
    cout.setf(std::ios::fixed); cout.precision(8);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            complexd v = A[idx(i, j, N)];
            // print as a+bi
            cout << "(" << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "i) ";
        }
        cout << "\n";
    }
    cout << "\n";
}

// ------ BLAS wrapper for C = alpha * A * B + beta * C, row-major, complex ------
void matmul(const CMatrix& A, const CMatrix& B, CMatrix& C,
    int N, complexd alpha = complexd(1.0, 0.0), complexd beta = complexd(0.0, 0.0)) {
    // cblas_zgemm expects pointers to double complex (std::complex<double> compatible)
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        &alpha,
        (const void*)A.data(), N,
        (const void*)B.data(), N,
        &beta,
        (void*)C.data(), N);
}

// ------ compute 1-norm (maximum column sum) of matrix A (complex) ------
double mat_one_norm(const CMatrix& A, int N) {
    double maxcol = 0.0;
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += std::abs(A[idx(i, j, N)]);
        if (sum > maxcol) maxcol = sum;
    }
    return maxcol;
}

// ------ add: C = A + B  ------
void mat_add(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
    size_t n = (size_t)N * N;
    for (size_t k = 0; k < n; ++k) C[k] = A[k] + B[k];
}

// ------ sub: C = A - B  ------
void mat_sub(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
    size_t n = (size_t)N * N;
    for (size_t k = 0; k < n; ++k) C[k] = A[k] - B[k];
}

// ------ scale in place: A *= alpha ------
void mat_scale_inplace(CMatrix& A, int N, complexd alpha) {
    size_t n = (size_t)N * N;
    for (size_t k = 0; k < n; ++k) A[k] *= alpha;
}

// ------ copy ------
CMatrix mat_copy(const CMatrix& A) { return CMatrix(A); }

// ------ solve linear system Q X = P  (both N x N), returns X in P_on_exit (overwrites) ------
int mat_solve(CMatrix& Q, CMatrix& P, int N) {
    // Use LAPACK zgetrf + zgetrs to solve multiple RHS. We'll solve Q*X = P.
    // Q and P are row-major; LAPACK expects row-major when using LAPACKE_... with row-major flag.
    vector<lapack_int> ipiv(N);
    lapack_int info;

    // LU factorization Q <- LU
    info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, N, N, (lapack_complex_double*)Q.data(), N, ipiv.data());
    if (info != 0) return info;

    // Solve Q*X = P (P is overwritten by X)
    info = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', N, N, (lapack_complex_double*)Q.data(), N, ipiv.data(),
        (lapack_complex_double*)P.data(), N);
    return info;
}

// ------ Pade(13) coefficients (Higham) ------
const double pade13_coeffs_d[] = {
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0
};

// ------ compute expm(A) using scaling & squaring + pade13, A is complex N x N ------

/*
┌──────────────┐
│   вход A     │
└──────┬───────┘
       ▼
вычислить норму A
       │
┌───── > если || A || > 5.4 ─┐
│                            ▼
│                      масштабировать A = A / 2 ^ s
│                            │
└────────────────────────────┘
▼
построить A², A⁴, A⁶
▼
вычислить U и V(Паде13)
▼
решить(V–U)X = (V + U)
▼
X = exp(A / 2 ^ s)
▼
возвести X в квадрат s раз
▼
результат e ^ A

*/
CMatrix expm_pade13(const CMatrix& A_in, int N) {
    // theta_13 value from Higham (to choose scaling)
    const double theta_13 = 5.4; // Higham suggests ~5.4 (we use 5.4)
    // Copy input
    CMatrix A = mat_copy(A_in);

    // norm(A,1)
    double A_norm = mat_one_norm(A, N);

    // determine s (number of scalings)
    int s = 0;
    if (A_norm > theta_13) {
        double val = A_norm / theta_13;
        s = std::max(0, (int)std::ceil(std::log2(val)));
    }

    // scale A <- A / 2^s
    if (s > 0) {
        double factor = 1.0 / std::pow(2.0, s);
        mat_scale_inplace(A, N, complexd(factor, 0.0));
    }

    // Precompute powers: A2 = A^2, A4 = A2^2, A6 = A4 * A2
    CMatrix A2((size_t)N * N, complexd(0.0, 0.0));
    matmul(A, A, A2, N);

    CMatrix A4((size_t)N * N, complexd(0.0, 0.0));
    matmul(A2, A2, A4, N);

    CMatrix A6((size_t)N * N, complexd(0.0, 0.0));
    matmul(A4, A2, A6, N);

    // Convert coefficients to complexd for convenience
    complexd c[14];
    for (int i = 0; i < 14; i++) c[i] = complexd(pade13_coeffs_d[i], 0.0);

    // Build U and V (see Higham / standard formula)
    // U = A*( A6*(c13*A6 + c11*A4 + c9*A2) + c7*A6 + c5*A4 + c3*A2 + c1*I )
    // V = A6*(c12*A6 + c10*A4 + c8*A2) + c6*A6 + c4*A4 + c2*A2 + c0*I

    CMatrix tmp((size_t)N * N, complexd(0.0, 0.0));
    CMatrix X((size_t)N * N, complexd(0.0, 0.0));
    CMatrix U((size_t)N * N, complexd(0.0, 0.0));
    CMatrix V((size_t)N * N, complexd(0.0, 0.0));
    CMatrix I = eye(N);

    // compute X = c13*A6 + c11*A4 + c9*A2
    // tmp = c13*A6
    for (size_t k = 0; k < tmp.size(); ++k) tmp[k] = A6[k] * c[0]; // careful: coeff array order: c[0]=6476... corresponds to highest? 
    // But the classical mapping uses c0..c13 with c0 largest (see coefficients defined above).
    // We'll follow the standard reconstruction below using explicit constants indices based on above pattern.

    // To avoid confusion, we follow the standard use:
    // b0 = pade13_coeffs_d[0], b1 = pade13_coeffs_d[1], ..., b13 = pade13_coeffs_d[13]
    // The formulas used below rely on that arrangement.

    // Build pieces step by step using the explicit pattern:
    // According to widely used implementation (Higham/Matlab/SciPy),
    // U = A*(A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I)
    // V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I

    // So mapping:
    // b0 = coeffs[0], b1 = coeffs[1], ..., b13 = coeffs[13]
    // Implement accordingly.

    // compute Z1 = b13*A6 + b11*A4 + b9*A2
    CMatrix Z1((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < Z1.size(); ++k)
        Z1[k] = A6[k] * complexd(pade13_coeffs_d[13], 0.0)   // b13
        + A4[k] * complexd(pade13_coeffs_d[11], 0.0) // b11
        + A2[k] * complexd(pade13_coeffs_d[9], 0.0); // b9

    // compute tmp = A6 * Z1  (matrix multiplication)
    CMatrix tmp1((size_t)N * N, complexd(0.0, 0.0));
    matmul(A6, Z1, tmp1, N);

    // now accumulate U_inner = tmp1 + b7*A6 + b5*A4 + b3*A2 + b1*I
    CMatrix U_inner((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < U_inner.size(); ++k)
        U_inner[k] = tmp1[k]
        + A6[k] * complexd(pade13_coeffs_d[7], 0.0)   // b7
        + A4[k] * complexd(pade13_coeffs_d[5], 0.0)   // b5
        + A2[k] * complexd(pade13_coeffs_d[3], 0.0)   // b3
        + I[k] * complexd(pade13_coeffs_d[1], 0.0);  // b1

    // U = A * U_inner
    matmul(A, U_inner, U, N);

    // V part:
    // Z2 = b12*A6 + b10*A4 + b8*A2
    CMatrix Z2((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < Z2.size(); ++k)
        Z2[k] = A6[k] * complexd(pade13_coeffs_d[12], 0.0)
        + A4[k] * complexd(pade13_coeffs_d[10], 0.0)
        + A2[k] * complexd(pade13_coeffs_d[8], 0.0);

    // tmp2 = A6 * Z2
    CMatrix tmp2((size_t)N * N, complexd(0.0, 0.0));
    matmul(A6, Z2, tmp2, N);

    // V = tmp2 + b6*A6 + b4*A4 + b2*A2 + b0*I
    for (size_t k = 0; k < V.size(); ++k)
        V[k] = tmp2[k]
        + A6[k] * complexd(pade13_coeffs_d[6], 0.0)
        + A4[k] * complexd(pade13_coeffs_d[4], 0.0)
        + A2[k] * complexd(pade13_coeffs_d[2], 0.0)
        + I[k] * complexd(pade13_coeffs_d[0], 0.0);

    // Now form P = V + U and Q = V - U, and solve Q X = P
    CMatrix P = V; mat_add(P, U, P, N);      // P = V + U
    CMatrix Q = V; mat_sub(Q, U, Q, N);      // Q = V - U

    // Solve Q * X = P  -> X = Q^{-1} * P
    int info = mat_solve(Q, P, N);
    if (info != 0) {
        cerr << "mat_solve (LU) failed with info=" << info << "\n";
        // Return something (identity) to avoid crash
        return eye(N);
    }
    CMatrix R = P; // R is result of P after solving (i.e. X)

    // now square: repeat R = R * R, s times
    for (int i = 0; i < s; i++) {
        CMatrix R2((size_t)N * N, complexd(0.0, 0.0));
        matmul(R, R, R2, N);
        R.swap(R2);
    }

    return R;
}

// ------ Example: A(t) for quantum-like problem
// We'll implement a typical complex Hamiltonian example:
// Let H(t) = [ [0, V*exp(i*omega*t)], [V*exp(-i*omega*t), 0] ]
// and set A(t) = -i * H(t) / hbar  (so A is anti-Hermitian -> exp is unitary)
CMatrix A_of_t(double t, int N) {
    // Example uses N==2
    assert(N == 2);
    const double hbar = 1.0;
    double V = 1.0;
    double omega = 0.3;
    complexd off = complexd(0.0, 1.0) * V * std::exp(complexd(0.0, omega * t)); // i*V*e^{i omega t}
    // But we want A = -i H / hbar, where H = [[0, V e^{iωt}], [V e^{-iωt}, 0]]
    // So A = -i * H = -i * [[0, w],[w*,0]] = [[0, -i w],[ -i w*, 0]] -> complex
    CMatrix M(4);
    complexd w = V * std::exp(complexd(0.0, omega * t)); // V e^{iωt}
    complexd minus_i = complexd(0.0, -1.0);
    M[idx(0, 0, N)] = complexd(0.0, 0.0);
    M[idx(0, 1, N)] = minus_i * w;           // -i * w
    M[idx(1, 0, N)] = minus_i * std::conj(w); // -i * w*
    M[idx(1, 1, N)] = complexd(0.0, 0.0);
    return M;
}

// ------ Magnus single step: U_next = exp(h * A(t + h/2)) * U
CMatrix magnus_step_complex(const CMatrix& U, double t, double h, int N) {
    CMatrix A_mid = A_of_t(t + 0.5 * h, N);
    // Omega = h * A_mid
    mat_scale_inplace(A_mid, N, complexd(h, 0.0));
    // expOmega
    CMatrix expOmega = expm_pade13(A_mid, N);
    // U_next = expOmega * U
    CMatrix Unext((size_t)N * N, complexd(0.0, 0.0));
    matmul(expOmega, U, Unext, N);
    return Unext;
}

// ------ Solver over [0,T] with nsteps ------
CMatrix magnus_solver_complex(double T, int nsteps, int N) {
    double h = T / nsteps;
    double t = 0.0;
    CMatrix U = eye(N);
    for (int k = 0; k < nsteps; k++) {
        U = magnus_step_complex(U, t, h, N);
        t += h;
    }
    return U;
}

double check_unitarity(const CMatrix& U, int N) {
    // Compute ||U†U - I||_F (норма Фробениуса)
    CMatrix Udag(N * N);
    // conj transpose
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Udag[idx(i, j, N)] = std::conj(U[idx(j, i, N)]);

    CMatrix UU(N * N, complexd(0, 0));
    matmul(Udag, U, UU, N);

    double diff = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            complexd target = (i == j) ? complexd(1.0, 0.0) : complexd(0.0, 0.0);
            diff += std::norm(UU[idx(i, j, N)] - target);
        }
    return std::sqrt(diff);
}

// ------ main: demo for N=2 ------
int main() {
    int N = 2;
    double T = 5.0;
    int steps = 200;

    CMatrix U = magnus_solver_complex(T, steps, N);
    printMat(U, N, "U(T)");

    double err = check_unitarity(U, N);
    std::cout << "Unitary check ||UtU - I||_F = " << err << std::endl;

    return 0;
}
