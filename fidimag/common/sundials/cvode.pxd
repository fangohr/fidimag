# -*- coding: utf-8 -*-
cdef extern from "sundials/sundials_types.h":
    ctypedef double realtype
    ctypedef bint booleantype

cdef extern from "sundials/sundials_nvector.h":
    cdef struct _generic_N_Vector:
        void *content
        
    ctypedef _generic_N_Vector *N_Vector
    N_Vector N_VNew_OpenMP(long int vec_length)
    void N_VDestroy_OpenMP(N_Vector v)
    void N_VPrint_OpenMP(N_Vector v)

cdef extern from "nvector/nvector_openmp.h":
    cdef N_Vector N_VMake_OpenMP(long int vec_length, realtype *v_data,
    int num_threads)
    
    cdef struct _N_VectorContent_OpenMP:
        long int length
        booleantype own_data
        realtype *data
        int num_threads
        
    ctypedef _N_VectorContent_OpenMP *N_VectorContent_OpenMP

	

cdef extern from "cvode/cvode.h":
    int CV_ADAMS
    int CV_BDF
    int CV_FUNCTIONAL
    int CV_NEWTON
    int CV_NORMAL
    int CV_ONE_STEP

    int CV_SUCCESS
    int CV_TSTOP_RETURN
    int CV_ROOT_RETURN

    int CV_WARNING

    int CV_TOO_MUCH_WORK
    int CV_TOO_MUCH_ACC
    int CV_ERR_FAILURE
    int CV_CONV_FAILURE

    int CV_LINIT_FAIL
    int CV_LSETUP_FAIL
    int CV_LSOLVE_FAIL
    int CV_RHSFUNC_FAIL
    int CV_FIRST_RHSFUNC_ERR
    int CV_REPTD_RHSFUNC_ERR
    int CV_UNREC_RHSFUNC_ERR
    int CV_RTFUNC_FAIL

    int CV_MEM_FAIL
    int CV_MEM_NULL
    int CV_ILL_INPUT
    int CV_NO_MALLOC
    int CV_BAD_K
    int CV_BAD_T
    int CV_BAD_DKY
    int CV_TOO_CLOSE
    
    ctypedef int (*CVRhsFn)(realtype t, N_Vector y, N_Vector ydot, void *user_data)
    ctypedef int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
    
    void *CVodeCreate(int lmm, int iter)
    int CVodeStep "CVode"(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask) nogil
    int CVodeSetUserData(void *cvode_mem, void *user_data)
    int CVodeSetMaxOrd(void *cvode_mem, int maxord)
    int CVodeSetMaxNumSteps(void *cvode_mem, long int mxsteps)
    int CVodeSetMaxHnilWarns(void *cvode_mem, int mxhnil)
    int CVodeSetStabLimDet(void *cvode_mem, booleantype stldet)
    int CVodeSetInitStep(void *cvode_mem, realtype hin)
    int CVodeSetMinStep(void *cvode_mem, realtype hmin)
    int CVodeSetMaxStep(void *cvode_mem, realtype hmax)
    int CVodeSetStopTime(void *cvode_mem, realtype tstop)
    int CVodeSetMaxErrTestFails(void *cvode_mem, int maxnef)
    int CVodeSetMaxNonlinIters(void *cvode_mem, int maxcor)
    int CVodeSetMaxConvFails(void *cvode_mem, int maxncf)
    int CVodeSetNonlinConvCoef(void *cvode_mem, realtype nlscoef)
    int CVodeSetIterType(void *cvode_mem, int iter)
    int CVodeSetRootDirection(void *cvode_mem, int *rootdir)
    int CVodeSetNoInactiveRootWarn(void *cvode_mem)
    int CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0)
    int CVodeReInit(void *cvode_mem, realtype t0, N_Vector y0)
    int CVodeSStolerances(void *cvode_mem, realtype reltol, realtype abstol)
    int CVodeSVtolerances(void *cvode_mem, realtype reltol, N_Vector abstol)
    int CVodeRootInit(void *cvode_mem, int nrtfn, CVRootFn g)
    int CVode(void *cvode_mem, realtype tout, N_Vector yout, realtype *tret, int itask)
    int CVodeGetDky(void *cvode_mem, realtype t, int k, N_Vector dky)
    int CVodeGetWorkSpace(void *cvode_mem, long int *lenrw, long int *leniw)
    int CVodeGetNumSteps(void *cvode_mem, long int *nsteps)
    int CVodeGetNumRhsEvals(void *cvode_mem, long int *nfevals)
    int CVodeGetNumLinSolvSetups(void *cvode_mem, long int *nlinsetups)
    int CVodeGetNumErrTestFails(void *cvode_mem, long int *netfails)
    int CVodeGetLastOrder(void *cvode_mem, int *qlast)
    int CVodeGetCurrentOrder(void *cvode_mem, int *qcur)
    int CVodeGetNumStabLimOrderReds(void *cvode_mem, long int *nslred)
    int CVodeGetActualInitStep(void *cvode_mem, realtype *hinused)
    int CVodeGetLastStep(void *cvode_mem, realtype *hlast)
    int CVodeGetCurrentStep(void *cvode_mem, realtype *hcur)
    int CVodeGetCurrentTime(void *cvode_mem, realtype *tcur)
    int CVodeGetTolScaleFactor(void *cvode_mem, realtype *tolsfac)
    int CVodeGetErrWeights(void *cvode_mem, N_Vector eweight)
    int CVodeGetEstLocalErrors(void *cvode_mem, N_Vector ele)
    int CVodeGetNumGEvals(void *cvode_mem, long int *ngevals)
    int CVodeGetRootInfo(void *cvode_mem, int *rootsfound)
    int CVodeGetIntegratorStats(void *cvode_mem, long int *nsteps,
                                long int *nfevals, long int *nlinsetups,
                                long int *netfails, int *qlast,
                                int *qcur, realtype *hinused, realtype *hlast,
                                realtype *hcur, realtype *tcur)
    int CVodeGetNumNonlinSolvIters(void *cvode_mem, long int *nniters)
    int CVodeGetNumNonlinSolvConvFails(void *cvode_mem, long int *nncfails)
    int CVodeGetNonlinSolvStats(void *cvode_mem, long int *nniters, long int *nncfails)
    char *CVodeGetReturnFlagName(int flag)
    void CVodeFree(void **cvode_mem)
    
    int CVDlsGetNumJacEvals(void *cvode_mem, long int *njevals)
    int CVDlsGetNumRhsEvals(void *cvode_mem, long int *nrevalsLS)
    int CVSpilsGetNumJtimesEvals(void *cvode_mem, long int *njevals)
	

cdef extern from "cvode/cvode_spgmr.h":
    int CVSpgmr(void *cvode_mem, int pretype, int max1)

cdef extern from "cvode/cvode_diag.h":
    int CVDiag(void *cvode_mem)

cdef extern from "cvode/cvode_spils.h":
    int CVSpilsSetPrecType(void *cvode_mem, int pretype)
    int CVSpilsSetGSType(void *cvode_mem, int gstype)
    int CVSpilsSetMaxl(void *cvode_mem, int maxl)
    int CVSpilsSetEpsLin(void *cvode_mem, realtype eplifac)

    ctypedef int (*CVSpilsPrecSetupFn)(realtype t, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype gamma, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);  
    ctypedef int (*CVSpilsPrecSolveFn)(realtype t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype gamma, realtype delta, int lr, void *user_data, N_Vector tmp); 
    int CVSpilsSetPreconditioner(void *cvode_mem, CVSpilsPrecSetupFn pset, CVSpilsPrecSolveFn psolve);

    ctypedef int (*CVSpilsJacTimesVecFn)(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector tmp)
    int CVSpilsSetJacTimesVecFn(void *cvode_mem, CVSpilsJacTimesVecFn jtv)
    
cdef extern from "sundials/sundials_iterative.h":
    int PREC_NONE
    int PREC_LEFT
    int PREC_RIGHT
    int PREC_BOTH
    
    int MODIFIED_GS
    int CLASSICAL_GS
