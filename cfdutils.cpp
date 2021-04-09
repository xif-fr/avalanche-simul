#include <inttypes.h>

extern "C" {
	
	#include "cfdutils.h"

	uint32_t build_FD_2D_DrhoD_matrix (uint32_t Nx_phys, uint32_t Ny_phys, double Δx, double Δy, const double* ρ_numpy, bool BCdir_left, bool BCdir_right, bool BCdir_top, bool BCdir_bot, uint32_t* row, uint32_t* col, double* val) {

		double Δx2 = Δx*Δx;
		double Δy2 = Δy*Δy;
		uint32_t N = Ny_phys;
		uint32_t k = 0;

		auto ρ = [&] (uint32_t i, uint32_t j) -> double {
			return ρ_numpy[ (i+1)*(Ny_phys+2)+(j+1) ];
		};

		for (uint32_t i = 0; i < Nx_phys; i++) {
			for (uint32_t j = 0; j < Ny_phys; j++) {

				uint32_t c = i*N+j;
				double vcenter = 0;

				if (j < Ny_phys-1) {
					row[k] = i*N+(j+1);
					col[k] = c;
					val[k] = 1/ρ(i,j) / Δy2;
					k++;
					vcenter += -1/ρ(i,j) / Δy2;   // ρ0
				} else if (BCdir_bot) {
					vcenter += -2/ρ(i,j) / Δy2;
				}

				if (j > 0) {
					row[k] = i*N+(j-1);
					col[k] = c;
					val[k] = 1/ρ(i,j-1) / Δy2;
					k++;
					vcenter += -1/ρ(i,j-1) / Δy2;   // ρ-
				} else if (BCdir_top) {
					vcenter += -2/ρ(i,j-1) / Δy2;
				}

				if (i < Nx_phys-1) {
					row[k] = (i+1)*N+j;
					col[k] = c;
					val[k] = 1/ρ(i,j) / Δx2;
					k++;
					vcenter += -1/ρ(i,j) / Δx2;   // ρ0
				} else if (BCdir_right) {
					vcenter += -2/ρ(i,j) / Δx2;
				}

				if (i > 0) {
					row[k] = (i-1)*N+j;
					col[k] = c;
					val[k] = 1/ρ(i-1,j) / Δx2;
					k++;
					vcenter += -1/ρ(i-1,j) / Δx2;   // ρ-
				} else if (BCdir_left) {
					vcenter += -1/ρ(i-1,j) / Δx2;
				}

				row[k] = i*N+j;
				col[k] = c;
				val[k] = vcenter;
				k++;
			}
		}
		return k;
	}

	uint32_t build_FD_2D_ILapAdv_centered_matrix (uint32_t Nx_phys, uint32_t Ny_phys, double Δx, double Δy, const double* ρ_numpy, const double* ρ_next_numpy, const double* u_numpy, const double* v_numpy, double µ, double Δt, bool BCdir_left, bool BCdir_right, bool BCdir_top, bool BCdir_bot, uint32_t* row, uint32_t* col, double* val) {

		double µ_Δx2 = µ/(Δx*Δx);
		double µ_Δy2 = µ/(Δy*Δy);
		uint32_t N = Ny_phys;
		uint32_t k = 0;

		auto ρ = [&] (uint32_t i, uint32_t j) -> double {
			return ρ_numpy[ (i+1)*(Ny_phys+2)+(j+1) ];
		};
		auto ρnext = [&] (uint32_t i, uint32_t j) -> double {
			return ρ_next_numpy[ (i+1)*(Ny_phys+2)+(j+1) ];
		};
		auto u = [&] (uint32_t i, uint32_t j) -> double {
			return u_numpy[ (i+1)*(Ny_phys+2)+(j+1) ];
		};
		auto v = [&] (uint32_t i, uint32_t j) -> double {
			return v_numpy[ (i+1)*(Ny_phys+2)+(j+1) ];
		};

		for (uint32_t i = 0; i < Nx_phys; i++) {
			for (uint32_t j = 0; j < Ny_phys; j++) {

				uint32_t c = i*N+j;
				double vcenter = ρ(i,j) / Δt;
				double ρu_2Δx = ρnext(i,j) * u(i,j) / 2 / Δy;
				double ρv_2Δy = ρnext(i,j) * v(i,j) / 2 / Δx;

				if (j < Ny_phys-1) {
					row[k] = i*N+(j+1);
					col[k] = c;
					val[k] = -µ_Δy2 +ρv_2Δy;
					k++;
					vcenter += µ_Δy2;
				} else {
					if (BCdir_bot) 
						vcenter += 2*µ_Δy2 + ρv_2Δy;
					else
						vcenter += -ρv_2Δy;
				}

				if (j > 0) {
					row[k] = i*N+(j-1);
					col[k] = c;
					val[k] = -µ_Δy2 -ρv_2Δy;
					k++;
					vcenter += µ_Δy2;
				} else {
					if (BCdir_top) 
						vcenter += 2*µ_Δy2 - ρv_2Δy;
					else
						vcenter += +ρv_2Δy;
				}

				if (i < Nx_phys-1) {
					row[k] = (i+1)*N+j;
					col[k] = c;
					val[k] = -µ_Δx2 +ρu_2Δx;
					k++;
					vcenter += µ_Δx2;
				} else {
					if (BCdir_right) 
						vcenter += 2*µ_Δx2 + ρu_2Δx;
					else
						vcenter += -ρu_2Δx;
				}

				if (i > 0) {
					row[k] = (i-1)*N+j;
					col[k] = c;
					val[k] = -µ_Δx2 -ρu_2Δx;
					k++;
					vcenter += µ_Δx2;
				} else {
					if (BCdir_left) 
						vcenter += 2*µ_Δx2 - ρu_2Δx;
					else
						vcenter += +ρu_2Δx;
				}

				row[k] = i*N+j;
				col[k] = c;
				val[k] = vcenter;
				k++;
			}
		}
		return k;
	}

}