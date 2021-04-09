
uint32_t build_FD_2D_DrhoD_matrix (uint32_t Nx_phys, uint32_t Ny_phys, double dx, double dy, const double* rho, bool BCdir_left, bool BCdir_right, bool BCdir_top, bool BCdir_bot, uint32_t* row, uint32_t* col, double* val);

uint32_t build_FD_2D_ILapAdv_centered_matrix (uint32_t Nx_phys, uint32_t Ny_phys, double dx, double dy, const double* rho, const double* rho_next, const double* u, const double* v, double mu, double dt, bool BCdir_left, bool BCdir_right, bool BCdir_top, bool BCdir_bot, uint32_t* row, uint32_t* col, double* val);