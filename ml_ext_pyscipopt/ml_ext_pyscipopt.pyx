
import pyscipopt
import numpy as np

# 定义一个类，继承 PySCIPOpt 的 Model 类，加一个 getKhalilState 方法
cdef class SCIP_ML_Model(pyscipopt.Model):
    def __init__(self, problemName='model', defaultPlugins=True, sourceModel=None, origcopy=False, globalcopy=True, enablepricing=False):
        super().__init__(problemName, defaultPlugins, sourceModel, origcopy, globalcopy, enablepricing)

    def getKhalilState(self, root_info, candidates):
        cdef SCIP* scip = self._scip

        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs
        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs_pos
        cdef np.ndarray[np.float32_t, ndim=1] cand_coefs_neg
        cdef np.ndarray[np.int32_t, ndim=1]   cand_nnzrs
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_cdeg_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_cdeg_var
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_cdeg_min
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_cdeg_max
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_pcoefs_count
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_var
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_pcoefs_max
        cdef np.ndarray[np.int32_t, ndim=1]   cand_root_ncoefs_count
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_var
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_root_ncoefs_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_slack
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_up
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_down
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_sum
        cdef np.ndarray[np.float32_t, ndim=1] cand_ps_product
        cdef np.ndarray[np.float32_t, ndim=1] cand_nb_up_infeas
        cdef np.ndarray[np.float32_t, ndim=1] cand_nb_down_infeas
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_mean
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_var
        cdef np.ndarray[np.int32_t, ndim=1]   cand_cdeg_min
        cdef np.ndarray[np.int32_t, ndim=1]   cand_cdeg_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_mean_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_min_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_cdeg_max_ratio
        cdef np.ndarray[np.float32_t, ndim=1] cand_prhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_prhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_nrhs_ratio_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_nrhs_ratio_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pp_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pp_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pn_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_pn_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_np_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_np_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_nn_max
        cdef np.ndarray[np.float32_t, ndim=1] cand_ota_nn_min
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_sum4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_mean4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_var4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_max4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_min4
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb1
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb2
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb3
        cdef np.ndarray[np.float32_t, ndim=1] cand_acons_nb4

        cdef int ncands = len(candidates)

        cand_coefs               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_coefs_pos           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_coefs_neg           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nnzrs               = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_cdeg_mean      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_cdeg_var       = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_cdeg_min       = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_cdeg_max       = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_pcoefs_count   = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_pcoefs_var     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_mean    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_min     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_pcoefs_max     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_count   = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_root_ncoefs_mean    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_var     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_min     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_root_ncoefs_max     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_solfracs            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_slack               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_up               = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_down             = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_ratio            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_sum              = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ps_product          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_frac_up_infeas      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_frac_down_infeas    = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nb_up_infeas        = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nb_down_infeas      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_mean           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_var            = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_min            = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_cdeg_max            = np.empty(shape=(ncands, ), dtype=np.int32)
        cand_cdeg_mean_ratio     = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_min_ratio      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_cdeg_max_ratio      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_prhs_ratio_max      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_prhs_ratio_min      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nrhs_ratio_max      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_nrhs_ratio_min      = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pp_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pp_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pn_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_pn_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_np_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_np_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_nn_max          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_ota_nn_min          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean1         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min1          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean2         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min2          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean3         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min3          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_sum4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_mean4         = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_var4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_max4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_min4          = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb1           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb2           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb3           = np.empty(shape=(ncands, ), dtype=np.float32)
        cand_acons_nb4           = np.empty(shape=(ncands, ), dtype=np.float32)

        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)
        cdef int i, cand_i, col_i

        # Static
        # ------
        cdef SCIP_ROW** neighbors
        cdef SCIP_Real* nonzero_coefs_raw
        cdef SCIP_Real* all_coefs_raw
        cdef SCIP_Real activity, lhs, rhs, coef
        cdef SCIP_VAR* var
        cdef SCIP_COL* col
        cdef int neighbor_index, cdeg_max, cdeg_min, cdeg, nb_neighbors
        cdef float cdeg_mean, cdeg_var
        cdef int pcoefs_count, ncoefs_count
        cdef float pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max
        cdef float ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max

        # if at root node, extract root information
        if SCIPgetNNodes(scip) == 1:
            root_info['col'] = {}
            root_info['col']['coefs']               = {}
            root_info['col']['coefs_pos']           = {}
            root_info['col']['coefs_neg']           = {}
            root_info['col']['nnzrs']               = {}
            root_info['col']['root_cdeg_mean']      = {}
            root_info['col']['root_cdeg_var']       = {}
            root_info['col']['root_cdeg_min']       = {}
            root_info['col']['root_cdeg_max']       = {}
            root_info['col']['root_pcoefs_count']   = {}
            root_info['col']['root_pcoefs_var']     = {}
            root_info['col']['root_pcoefs_mean']    = {}
            root_info['col']['root_pcoefs_min']     = {}
            root_info['col']['root_pcoefs_max']     = {}
            root_info['col']['root_ncoefs_count']   = {}
            root_info['col']['root_ncoefs_mean']    = {}
            root_info['col']['root_ncoefs_var']     = {}
            root_info['col']['root_ncoefs_min']     = {}
            root_info['col']['root_ncoefs_max']     = {}
            for i in range(ncols):
                col = cols[i]
                col_i = SCIPcolGetIndex(col)
                neighbors = SCIPcolGetRows(col)
                nb_neighbors = SCIPcolGetNNonz(col)
                nonzero_coefs_raw = SCIPcolGetVals(col)

                # Objective function coeffs. (3)
                #   Value of the coefficient (raw, positive only, negative only)
                root_info['col']['coefs'][col_i]     = SCIPcolGetObj(col)
                root_info['col']['coefs_pos'][col_i] = max(root_info['col']['coefs'][col_i], 0)
                root_info['col']['coefs_neg'][col_i] = min(root_info['col']['coefs'][col_i], 0)

                # Num. constraints (1)
                #   Number of constraints that the variable participates in (with a non-zero coefficient)
                root_info['col']['nnzrs'][col_i] = nb_neighbors

                # Stats. for constraint degrees (4)
                #   The degree of a constraint is the number of variables that participate in it. A variable may
                #   participate in multiple constraints, and statistics over those constraints’ degrees are used.
                #   The constraint degree is computed on the root LP (mean, stdev., min, max)
                cdeg_var, cdeg_mean, cdeg_min, cdeg_max = 0, 0, 0, 0
                if nb_neighbors > 0:
                    for neighbor_index in range(nb_neighbors):
                        cdeg = SCIProwGetNNonz(neighbors[neighbor_index])
                        cdeg_mean += cdeg
                        cdeg_max = cdeg if neighbor_index == 0 else max(cdeg_max, cdeg)
                        cdeg_min = cdeg if neighbor_index == 0 else min(cdeg_min, cdeg)
                    cdeg_mean /= nb_neighbors
                    for neighbor_index in range(nb_neighbors):
                        cdeg_var += (cdeg - cdeg_mean)**2
                    cdeg_var /= nb_neighbors
                root_info['col']['root_cdeg_mean'][col_i] = cdeg_mean
                root_info['col']['root_cdeg_var'][col_i] = cdeg_var
                root_info['col']['root_cdeg_min'][col_i] = cdeg_min
                root_info['col']['root_cdeg_max'][col_i] = cdeg_max

                # Stats. for constraint coeffs. (10)
                #   A variable’s positive (negative) coefficients in the constraints it participates in
                #   (count, mean, stdev., min, max)
                pcoefs_var, pcoefs_mean, pcoefs_min, pcoefs_max = 0, 0, 0, 0.
                ncoefs_var, ncoefs_mean, ncoefs_min, ncoefs_max = 0, 0, 0, 0.
                pcoefs_count, ncoefs_count = 0, 0
                for neighbor_index in range(nb_neighbors):
                    coef = nonzero_coefs_raw[neighbor_index]
                    if coef > 0:
                        pcoefs_count += 1
                        pcoefs_mean = coef
                        pcoefs_min = coef if pcoefs_count == 1 else min(pcoefs_min, coef)
                        pcoefs_max = coef if pcoefs_count == 1 else max(pcoefs_max, coef)
                    if coef < 0:
                        ncoefs_count += 1
                        ncoefs_mean += coef
                        ncoefs_min = coef if ncoefs_count == 1 else min(ncoefs_min, coef)
                        ncoefs_max = coef if ncoefs_count == 1 else max(ncoefs_max, coef)
                if pcoefs_count > 0:
                    pcoefs_mean /= pcoefs_count
                if ncoefs_count > 0:
                    ncoefs_mean /= ncoefs_count
                for neighbor_index in range(nb_neighbors):
                    coef = nonzero_coefs_raw[neighbor_index]
                    if coef > 0:
                        pcoefs_var += (coef - pcoefs_mean)**2
                    if coef < 0:
                        ncoefs_var += (coef - ncoefs_mean)**2
                if pcoefs_count > 0:
                    pcoefs_var /= pcoefs_count
                if ncoefs_count > 0:
                    ncoefs_var /= ncoefs_count
                root_info['col']['root_pcoefs_count'][col_i] = pcoefs_count
                root_info['col']['root_pcoefs_mean'][col_i]  = pcoefs_mean
                root_info['col']['root_pcoefs_var'][col_i]   = pcoefs_var
                root_info['col']['root_pcoefs_min'][col_i]   = pcoefs_min
                root_info['col']['root_pcoefs_max'][col_i]   = pcoefs_max
                root_info['col']['root_ncoefs_count'][col_i] = ncoefs_count
                root_info['col']['root_ncoefs_mean'][col_i]  = ncoefs_mean
                root_info['col']['root_ncoefs_var'][col_i]   = ncoefs_var
                root_info['col']['root_ncoefs_min'][col_i]   = ncoefs_min
                root_info['col']['root_ncoefs_max'][col_i]   = ncoefs_max

        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            col_i = SCIPcolGetIndex(col)
            cand_coefs[cand_i]             = root_info['col']['coefs'][col_i]
            cand_coefs_pos[cand_i]         = root_info['col']['coefs_pos'][col_i]
            cand_coefs_neg[cand_i]         = root_info['col']['coefs_neg'][col_i]
            cand_nnzrs[cand_i]             = root_info['col']['nnzrs'][col_i]
            cand_root_cdeg_mean[cand_i]    = root_info['col']['root_cdeg_mean'][col_i]
            cand_root_cdeg_var[cand_i]     = root_info['col']['root_cdeg_var'][col_i]
            cand_root_cdeg_min[cand_i]     = root_info['col']['root_cdeg_min'][col_i]
            cand_root_cdeg_max[cand_i]     = root_info['col']['root_cdeg_max'][col_i]
            cand_root_pcoefs_count[cand_i] = root_info['col']['root_pcoefs_count'][col_i]
            cand_root_pcoefs_mean[cand_i]  = root_info['col']['root_pcoefs_mean'][col_i]
            cand_root_pcoefs_var[cand_i]   = root_info['col']['root_pcoefs_var'][col_i]
            cand_root_pcoefs_min[cand_i]   = root_info['col']['root_pcoefs_min'][col_i]
            cand_root_pcoefs_max[cand_i]   = root_info['col']['root_pcoefs_max'][col_i]
            cand_root_ncoefs_count[cand_i] = root_info['col']['root_ncoefs_count'][col_i]
            cand_root_ncoefs_mean[cand_i]  = root_info['col']['root_ncoefs_mean'][col_i]
            cand_root_ncoefs_var[cand_i]   = root_info['col']['root_ncoefs_var'][col_i]
            cand_root_ncoefs_min[cand_i]   = root_info['col']['root_ncoefs_min'][col_i]
            cand_root_ncoefs_max[cand_i]   = root_info['col']['root_ncoefs_max'][col_i]

        # Simple dynamic
        # --------------
        cdef int neighbor_column_index, neighbor_ncolumns
        cdef float solval, pos_coef_sum, neg_coef_sum, neighbor_coef
        cdef float ota_pp_max, ota_pp_min, ota_pn_max, ota_pn_min
        cdef float ota_np_max, ota_np_min, ota_nn_max, ota_nn_min
        cdef float prhs_ratio_max, prhs_ratio_min
        cdef float nrhs_ratio_max, nrhs_ratio_min
        cdef float value, pratio, nratio
        cdef SCIP_VAR* neighbor_var
        cdef SCIP_Real* neighbor_columns_values
        cdef int nbranchings
        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)

            # Slack and ceil distances (2)
            #   min{xij−floor(xij),ceil(xij) −xij} and ceil(xij) −xij
            solval = SCIPcolGetPrimsol(col)
            cand_solfracs[cand_i] = SCIPfeasFrac(scip, solval)
            cand_slack[cand_i] = min(cand_solfracs[cand_i], 1-cand_solfracs[cand_i])

            # Pseudocosts (5)
            #   Upwards and downwards values, and their corresponding ratio, sum and product,
            #   weighted by the fractionality of xj
            cand_ps_up[cand_i] = SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS)
            cand_ps_down[cand_i] = SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS)
            cand_ps_sum[cand_i] = cand_ps_up[cand_i] + cand_ps_down[cand_i]
            cand_ps_ratio[cand_i] = 0 if cand_ps_up[cand_i] == 0 else cand_ps_up[cand_i] / cand_ps_sum[cand_i]
            cand_ps_product[cand_i] = cand_ps_up[cand_i] * cand_ps_down[cand_i]

            # Infeasibility statistics (4)
            #   Number and fraction of nodes for which applying SB to variable xj led to one (two)
            #   infeasible children (during data collection)
            # N.B. replaced by left, right infeasibility
            cand_nb_up_infeas[cand_i]   = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS)
            cand_nb_down_infeas[cand_i] = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS)
            nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS)
            cand_frac_up_infeas[cand_i]   = 0 if nbranchings == 0 else cand_nb_up_infeas[cand_i] / nbranchings
            nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS)
            cand_frac_down_infeas[cand_i] = 0 if nbranchings == 0 else cand_nb_down_infeas[cand_i] / nbranchings

            # Stats. for constraint degrees (7)
            #   A dynamic variant of the static version above. Here, the constraint degrees are
            #   on the current node’s LP.The ratios of the static mean, maximum and minimum to
            #   their dynamic counterparts are also features
            cdeg_var, cdeg_mean, cdeg_min, cdeg_max = 0, 0, 0, 0
            if nb_neighbors > 0:
                for neighbor_index in range(nb_neighbors):
                    cdeg = SCIProwGetNLPNonz(neighbors[neighbor_index])
                    cdeg_mean += cdeg
                    cdeg_max = cdeg if neighbor_index == 0 else max(cdeg_max, cdeg)
                    cdeg_min = cdeg if neighbor_index == 0 else min(cdeg_min, cdeg)
                cdeg_mean /= nb_neighbors
                for neighbor_index in range(nb_neighbors):
                    cdeg = SCIProwGetNLPNonz(neighbors[neighbor_index])
                    cdeg_var += (cdeg - cdeg_mean)**2
                cdeg_var /= nb_neighbors
            cand_cdeg_mean[cand_i] = cdeg_mean
            cand_cdeg_var[cand_i]  = cdeg_var
            cand_cdeg_min[cand_i]  = cdeg_min
            cand_cdeg_max[cand_i]  = cdeg_max
            cand_cdeg_mean_ratio[cand_i] = 0 if cdeg_mean == 0 else cdeg_mean / (cand_root_cdeg_mean[cand_i] + cdeg_mean)
            cand_cdeg_min_ratio[cand_i]  = 0 if cdeg_min == 0 else cdeg_min / (cand_root_cdeg_min[cand_i] + cdeg_min)
            cand_cdeg_max_ratio[cand_i]  = 0 if cdeg_max == 0 else cdeg_max / (cand_root_cdeg_max[cand_i] + cdeg_max)

            # Min/max for ratios of constraint coeffs. to RHS (4)
            #   Minimum and maximum ratios across positive and negative right-hand-sides (RHS)
            prhs_ratio_max, prhs_ratio_min = -1, 1
            nrhs_ratio_max, nrhs_ratio_min = -1, 1
            for neighbor_index in range(nb_neighbors):
                coef = nonzero_coefs_raw[neighbor_index]
                rhs = SCIProwGetRhs(neighbors[neighbor_index])
                lhs = SCIProwGetLhs(neighbors[neighbor_index])
                if not SCIPisInfinity(scip, REALABS(rhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(rhs))
                    if rhs >= 0:
                        rhs_ratio_max = max(prhs_ratio_max, value)
                        rhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)
                if not SCIPisInfinity(scip, REALABS(lhs)):
                    value = 0 if coef == 0 else coef / (REALABS(coef) + REALABS(lhs))
                    if -lhs >= 0:
                        prhs_ratio_max = max(prhs_ratio_max, value)
                        prhs_ratio_min = min(prhs_ratio_min, value)
                    else:
                        nrhs_ratio_max = max(nrhs_ratio_max, value)
                        nrhs_ratio_min = min(nrhs_ratio_min, value)
            cand_prhs_ratio_max[cand_i] = prhs_ratio_max
            cand_prhs_ratio_min[cand_i] = prhs_ratio_min
            cand_nrhs_ratio_max[cand_i] = nrhs_ratio_max
            cand_nrhs_ratio_min[cand_i] = nrhs_ratio_min

            # Min/max for one-to-all coefficient ratios (8)
            #   The statistics are over the ratios of a variable’s coefficient, to the sum over all
            #   other variables’ coefficients, for a given constraint. Four versions of these ratios
            #   are considered: positive (negative) coefficient to sum of positive (negative) coefficients
            ota_pp_max, ota_pp_min, ota_pn_max, ota_pn_min = 0, 1, 0, 1
            ota_np_max, ota_np_min, ota_nn_max, ota_nn_min = 0, 1, 0, 1
            for neighbor_index in range(nb_neighbors):
                all_coefs_raw = SCIProwGetVals(neighbors[neighbor_index])
                neighbor_ncolumns = SCIProwGetNNonz(neighbors[neighbor_index])
                pos_coef_sum, neg_coef_sum = 0, 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    neighbor_coef = all_coefs_raw[neighbor_column_index]
                    if neighbor_coef > 0:
                        pos_coef_sum += neighbor_coef
                    else:
                        neg_coef_sum += neighbor_coef
                coef = nonzero_coefs_raw[neighbor_index]
                if coef > 0:
                    pratio = coef / pos_coef_sum
                    nratio = coef / (coef - neg_coef_sum)
                    ota_pp_max = max(ota_pp_max, pratio)
                    ota_pp_min = min(ota_pp_min, pratio)
                    ota_pn_max = max(ota_pn_max, nratio)
                    ota_pn_min = min(ota_pn_min, nratio)
                if coef < 0:
                    pratio = coef / (coef - pos_coef_sum)
                    nratio = coef / neg_coef_sum
                    ota_np_max = max(ota_np_max, pratio)
                    ota_np_min = min(ota_np_min, pratio)
                    ota_nn_max = max(ota_nn_max, nratio)
                    ota_nn_min = min(ota_nn_min, nratio)
            cand_ota_pp_max[cand_i] = ota_pp_max
            cand_ota_pp_min[cand_i] = ota_pp_min
            cand_ota_pn_max[cand_i] = ota_pn_max
            cand_ota_pn_min[cand_i] = ota_pn_min
            cand_ota_np_max[cand_i] = ota_np_max
            cand_ota_np_min[cand_i] = ota_np_min
            cand_ota_nn_max[cand_i] = ota_nn_max
            cand_ota_nn_min[cand_i] = ota_nn_min

        # Active dynamic
        # --------------
        # Stats. for active constraint coefficients (24)
        #   An active constraint at a node LP is one which is binding with equality at the optimum.
        #   We consider 4 weighting schemes for an active constraint: unit weight, inverse of the
        #   sum of the coefficients of all variables in constraint, inverse of the sum of the coefficients
        #   of only candidate variables in constraint, dual cost of the constraint. Given the absolute
        #   value of the coefficients of xj in the active constraints, we compute the sum, mean, stdev.,
        #   max. and min. of those values, for each of the weighting schemes. We also compute the weighted
        #   number of active constraints that xj is in, with the same 4 weightings
        cdef int row_index
        cdef int nrows = SCIPgetNLPRows(scip)
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef float constraint_sum, abs_coef
        cdef SCIP_COL** neighbor_columns
        cdef int neighbor_var_index, candidate_index
        cdef int active_count
        cdef float acons_sum1, acons_mean1, acons_var1, acons_max1, acons_min1
        cdef float acons_sum2, acons_mean2, acons_var2, acons_max2, acons_min2
        cdef float acons_sum3, acons_mean3, acons_var3, acons_max3, acons_min3
        cdef float acons_sum4, acons_mean4, acons_var4, acons_max4, acons_min4
        cdef float acons_nb1, acons_nb2, acons_nb3, acons_nb4
        cdef np.ndarray[np.float32_t, ndim=1] act_cons_w1, act_cons_w2, act_cons_w3, act_cons_w4

        act_cons_w1 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w2 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w3 = np.zeros(shape=(nrows, ), dtype=np.float32)
        act_cons_w4 = np.zeros(shape=(nrows, ), dtype=np.float32)
        for row_index in range(nrows):
            row = rows[row_index]
            rhs = SCIProwGetRhs(row)
            lhs = SCIProwGetLhs(row)
            activity = SCIPgetRowActivity(scip, row)
            # N.B. active if activity = lhs or rhs
            if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                neighbor_columns = SCIProwGetCols(row)
                neighbor_ncolumns = SCIProwGetNNonz(row)
                neighbor_columns_values = SCIProwGetVals(row)

                # weight no. 1
                # unit weight
                act_cons_w1[row_index] = 1

                # weight no. 2
                # inverse of the sum of the coefficients of all variables in constraint
                constraint_sum = 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    constraint_sum += REALABS(neighbor_columns_values[neighbor_column_index])
                act_cons_w2[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum

                # weight no. 3
                # inverse of the sum of the coefficients of only candidate variables in constraint
                constraint_sum = 0
                for neighbor_column_index in range(neighbor_ncolumns):
                    neighbor_var = SCIPcolGetVar(neighbor_columns[neighbor_column_index])
                    neighbor_var_index = SCIPvarGetIndex(neighbor_var)
                    for cand_i in range(ncands):
                        var = (<Variable>candidates[cand_i]).scip_var
                        if SCIPvarGetIndex(var) == neighbor_var_index:
                            constraint_sum += REALABS(neighbor_columns_values[neighbor_column_index])
                            break
                act_cons_w3[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum

                # weight no. 4
                # dual cost of the constraint
                act_cons_w4[row_index] = REALABS(SCIProwGetDualsol(row))

        for cand_i in range(ncands):
            var = (<Variable>candidates[cand_i]).scip_var
            col = SCIPvarGetCol(var)
            neighbors = SCIPcolGetRows(col)
            nb_neighbors = SCIPcolGetNNonz(col)
            nonzero_coefs_raw = SCIPcolGetVals(col)

            acons_sum1, acons_mean1, acons_var1, acons_max1, acons_min1 = 0, 0, 0, 0, 0
            acons_sum2, acons_mean2, acons_var2, acons_max2, acons_min2 = 0, 0, 0, 0, 0
            acons_sum3, acons_mean3, acons_var3, acons_max3, acons_min3 = 0, 0, 0, 0, 0
            acons_sum4, acons_mean4, acons_var4, acons_max4, acons_min4 = 0, 0, 0, 0, 0
            acons_nb1,  acons_nb2,   acons_nb3,   acons_nb4             = 0, 0, 0, 0
            active_count = 0
            for neighbor_index in range(nb_neighbors):
                rhs = SCIProwGetRhs(neighbors[neighbor_index])
                lhs = SCIProwGetLhs(neighbors[neighbor_index])
                activity = SCIPgetRowActivity(scip, neighbors[neighbor_index])
                # N.B. active if activity = lhs or rhs
                if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                    active_count += 1
                    neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                    abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                    acons_nb1 += act_cons_w1[neighbor_row_index]
                    value = act_cons_w1[neighbor_row_index] * abs_coef
                    acons_sum1 += value
                    acons_max1 = value if active_count == 1 else max(acons_max1, value)
                    acons_min1 = value if active_count == 1 else min(acons_min1, value)

                    acons_nb2 += act_cons_w2[neighbor_row_index]
                    value = act_cons_w2[neighbor_row_index] * abs_coef
                    acons_sum2 += value
                    acons_max2 = value if active_count == 1 else max(acons_max2, value)
                    acons_min2 = value if active_count == 1 else min(acons_min2, value)

                    acons_nb3 += act_cons_w3[neighbor_row_index]
                    value = act_cons_w3[neighbor_row_index] * abs_coef
                    acons_sum3 += value
                    acons_max3 = value if active_count == 1 else max(acons_max3, value)
                    acons_min3 = value if active_count == 1 else min(acons_min3, value)

                    acons_nb4 += act_cons_w4[neighbor_row_index]
                    value = act_cons_w4[neighbor_row_index] * abs_coef
                    acons_sum4 += value
                    acons_max4 = value if active_count == 1 else max(acons_max4, value)
                    acons_min4 = value if active_count == 1 else min(acons_min4, value)

            if active_count > 0:
                acons_mean1 = acons_sum1 / active_count
                acons_mean2 = acons_sum2 / active_count
                acons_mean3 = acons_sum3 / active_count
                acons_mean4 = acons_sum4 / active_count
                for neighbor_index in range(nb_neighbors):
                    rhs = SCIProwGetRhs(neighbors[neighbor_index])
                    lhs = SCIProwGetLhs(neighbors[neighbor_index])
                    activity = SCIPgetRowActivity(scip, neighbors[neighbor_index])
                    # N.B. active if activity = lhs or rhs
                    if SCIPisEQ(scip, activity, rhs) or SCIPisEQ(scip, activity, lhs):
                        neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index])
                        abs_coef = REALABS(nonzero_coefs_raw[neighbor_index])

                        value = act_cons_w1[neighbor_row_index] * abs_coef
                        acons_var1 += (value - acons_mean1)**2

                        value = act_cons_w2[neighbor_row_index] * abs_coef
                        acons_var2 += (value - acons_mean2)**2

                        value = act_cons_w3[neighbor_row_index] * abs_coef
                        acons_var3 += (value - acons_mean3)**2

                        value = act_cons_w4[neighbor_row_index] * abs_coef
                        acons_var4 += (value - acons_mean4)**2
                acons_var1 /= active_count
                acons_var2 /= active_count
                acons_var3 /= active_count
                acons_var4 /= active_count

            cand_acons_sum1[cand_i]  = acons_sum1
            cand_acons_sum2[cand_i]  = acons_sum2
            cand_acons_sum3[cand_i]  = acons_sum3
            cand_acons_sum4[cand_i]  = acons_sum4
            cand_acons_mean1[cand_i] = acons_mean1
            cand_acons_mean2[cand_i] = acons_mean2
            cand_acons_mean3[cand_i] = acons_mean3
            cand_acons_mean4[cand_i] = acons_mean4
            cand_acons_max1[cand_i]  = acons_max1
            cand_acons_max2[cand_i]  = acons_max2
            cand_acons_max3[cand_i]  = acons_max3
            cand_acons_max4[cand_i]  = acons_max4
            cand_acons_min1[cand_i]  = acons_min1
            cand_acons_min2[cand_i]  = acons_min2
            cand_acons_min3[cand_i]  = acons_min3
            cand_acons_min4[cand_i]  = acons_min4
            cand_acons_var1[cand_i]  = acons_var1
            cand_acons_var2[cand_i]  = acons_var2
            cand_acons_var3[cand_i]  = acons_var3
            cand_acons_var4[cand_i]  = acons_var4
            cand_acons_nb1[cand_i]   = acons_nb1
            cand_acons_nb2[cand_i]   = acons_nb2
            cand_acons_nb3[cand_i]   = acons_nb3
            cand_acons_nb4[cand_i]   = acons_nb4

        return {
            'coefs':                cand_coefs,
            'coefs_pos':            cand_coefs_pos,
            'coefs_neg':            cand_coefs_neg,
            'nnzrs':                cand_nnzrs,
            'root_cdeg_mean':       cand_root_cdeg_mean,
            'root_cdeg_var':        cand_root_cdeg_var,
            'root_cdeg_min':        cand_root_cdeg_min,
            'root_cdeg_max':        cand_root_cdeg_max,
            'root_pcoefs_count':    cand_root_pcoefs_count,
            'root_pcoefs_mean':     cand_root_pcoefs_mean,
            'root_pcoefs_var':      cand_root_pcoefs_var,
            'root_pcoefs_min':      cand_root_pcoefs_min,
            'root_pcoefs_max':      cand_root_pcoefs_max,
            'root_ncoefs_count':    cand_root_ncoefs_count,
            'root_ncoefs_mean':     cand_root_ncoefs_mean,
            'root_ncoefs_var':      cand_root_ncoefs_var,
            'root_ncoefs_min':      cand_root_ncoefs_min,
            'root_ncoefs_max':      cand_root_ncoefs_max,
            'solfracs':             cand_solfracs,
            'slack':                cand_slack,
            'ps_up':                cand_ps_up,
            'ps_down':              cand_ps_down,
            'ps_ratio':             cand_ps_ratio,
            'ps_sum':               cand_ps_sum,
            'ps_product':           cand_ps_product,
            'nb_up_infeas':         cand_nb_up_infeas,
            'nb_down_infeas':       cand_nb_down_infeas,
            'frac_up_infeas':       cand_frac_up_infeas,
            'frac_down_infeas':     cand_frac_down_infeas,
            'cdeg_mean':            cand_cdeg_mean,
            'cdeg_var':             cand_cdeg_var,
            'cdeg_min':             cand_cdeg_min,
            'cdeg_max':             cand_cdeg_max,
            'cdeg_mean_ratio':      cand_cdeg_mean_ratio,
            'cdeg_min_ratio':       cand_cdeg_min_ratio,
            'cdeg_max_ratio':       cand_cdeg_max_ratio,
            'prhs_ratio_max':       cand_prhs_ratio_max,
            'prhs_ratio_min':       cand_prhs_ratio_min,
            'nrhs_ratio_max':       cand_nrhs_ratio_max,
            'nrhs_ratio_min':       cand_nrhs_ratio_min,
            'ota_pp_max':           cand_ota_pp_max,
            'ota_pp_min':           cand_ota_pp_min,
            'ota_pn_max':           cand_ota_pn_max,
            'ota_pn_min':           cand_ota_pn_min,
            'ota_np_max':           cand_ota_np_max,
            'ota_np_min':           cand_ota_np_min,
            'ota_nn_max':           cand_ota_nn_max,
            'ota_nn_min':           cand_ota_nn_min,
            'acons_sum1':           cand_acons_sum1,
            'acons_mean1':          cand_acons_mean1,
            'acons_var1':           cand_acons_var1,
            'acons_max1':           cand_acons_max1,
            'acons_min1':           cand_acons_min1,
            'acons_sum2':           cand_acons_sum2,
            'acons_mean2':          cand_acons_mean2,
            'acons_var2':           cand_acons_var2,
            'acons_max2':           cand_acons_max2,
            'acons_min2':           cand_acons_min2,
            'acons_sum3':           cand_acons_sum3,
            'acons_mean3':          cand_acons_mean3,
            'acons_var3':           cand_acons_var3,
            'acons_max3':           cand_acons_max3,
            'acons_min3':           cand_acons_min3,
            'acons_sum4':           cand_acons_sum4,
            'acons_mean4':          cand_acons_mean4,
            'acons_var4':           cand_acons_var4,
            'acons_max4':           cand_acons_max4,
            'acons_min4':           cand_acons_min4,
            'acons_nb1':            cand_acons_nb1,
            'acons_nb2':            cand_acons_nb2,
            'acons_nb3':            cand_acons_nb3,
            'acons_nb4':            cand_acons_nb4,
        }