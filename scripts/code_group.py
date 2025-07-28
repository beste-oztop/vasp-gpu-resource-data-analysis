#!/usr/bin/env python3

import re

class code_group:
  def __init__( self, name, regexp, gpu_status, algo_group ):
    self.name = name
    self.regexp = regexp
    self.gpu_status = gpu_status
    self.algo_group = algo_group
    self.pattern = [ re.compile( v, re.IGNORECASE ) for v in [name] + regexp ]
  #end code_group.__init__
#end class code group

def match_group(cmd):
    # Ensure cmd is a string, handle NaN/None gracefully
    if not isinstance(cmd, str):
        cmd = str(cmd) if cmd is not None else ""
    M = [CODE_GROUPS_D[k] for k in CODE_GROUPS_D if k.match(cmd)]
    if len(M):
        g = M[0]
    else:
        g = code_group(cmd, [cmd], "unknown", "unknown")
    return g
#match_group

def match_name( cmd ):
  g = match_group( cmd )
  return g.name

def match_algo( cmd ):
  g = match_group( cmd )
  return g.algo_group

def match( cmd ):
  return match_name( cmd )

CODE_GROUPS = [

  code_group("ATLAS", ["shifter"], "unlikely", "HEP"),

  code_group("VASP", ["gvasp"], "enabled", "DFT"),
  
  code_group("Amber", ["SANDER","PMEMD"], "enabled", "MD" ),

  code_group("AMD",   ["GROW-BUBBLES"],   "unknown", "MD" ),
  
  code_group("AmrDrv", [],        "unknown", "AMR" ),
  
  #dark-matter - Borrill
  code_group("ART", ["^ART-.*","^ART$"], "unknown", "Analytics"),
  
  code_group("blast", ["TF_WORKER", "parallel_blast.py"], "unknown", "BioInformatics"),
  
  code_group("AORSA", ["xaorsa", ".*_AORSA" ], "unknown", "Fusion-Cont."),
  
  code_group("amr3d", [], "unknown", "AMR"),
  
  code_group("BerkeleyGW", 
             ["xi0","XI0",".*\.REAL\.X",".*\.CPLX\.X","^SIGMA$",
              "SIGMA\.MPI","XCTON","^DIAG$"], 
             "unknown", "DFT"),
  
  code_group("cactus", [], "unknown", "AMR"),
  
  code_group("castro", [], "unknown", "AMR"),
  
  code_group("CESM", ["CCSM","^CAM$","^POP$", "cism", 
                      "init_atmosphere_model", 
                      "run_CESM", "run_CanESM2",
                      "run_GFDL", "run_HadGEM2", 
                      "run_MPI_ESM"], "Unlikely", "Climate"),
  
  code_group("charmm", [], "Enabled", "MD"),
  
  #tasks assignment based on tracking these jobs to user redwards
  code_group("chroma", ["CHROMA","^HMC","^RUN_NN_W",
                        "^tasks","gr_chroma", "wm_chroma"], 
             "enabled", "QCD"),
  
  code_group("Compo_Analysis", 
             ["global_fcst", "GLOBAL_FCST","GLOBAL_CYCLEP",
              "GLOBAL_POSTGPP","GLOBAL_ENKF"], 
             "unlikely", "Climate"),
  
  code_group("ChomboCrunch", 
             ["MOVINGCRUNCH","VISCOELASTICDRIVER","VISCOUSDRIVER",
              "viscoelasticTopeDriver", "driver-public2d" ], 
             "unknown", "AMR"),
  
  code_group("cpmd", [], "unknown", "DFT"),
  
  code_group("cp2k", [], "unknown", "DFT"),
  
  code_group("CS267", ["^BENCHMARK-NB"], "unknown", "unknown"),
  
  code_group("Cray_CCM", ["CCMLAUNCH"], "unknown", "unknown"),
  
  code_group("diag", [], "unknown", "unknown"),
  
  code_group("DLPOLY", [], "unknown", "MD"),
  
  code_group("elm_6", ["ELM_6"], "unknown", "unknown"),
  
  #electromagnetic wave inversion (seismic)
  code_group("EMGeo", ["EM3D_INV"], "unknown", "Seismic"),
  
  #electromagnetic wave inversion (seismic)
  code_group("EWI3D", ["HP_INV","ED_INV"], "unknown", "Seismic"),
  
  code_group("Espresso", 
             ["^PW.*\.X","^PH.*\.X","^CP.*\.X","^EPW.*\.X",
              "^neb.*\.x", "pp.*\.x", "dynmat.*\.x", 
              "turbo_lanczos.*\.x", "turbo_spectrum.*\.x", "turbo_davidson.*\.x",
              "xspectra.*\.x", "dos.*\.x", "bands.*\.x", "projwfc.*\.x",
              "cppp.*\.x", "pw_export.*\.x", "gipaw.*\.x" ], 
             "Enabled", "DFT"),
 
  code_group("Exes", [], "unknown", "unknown"),
  
  code_group("flash", [], "unknown", "AMR"),
  
  code_group("Gene", ["gene_hopper","gene_edison"], "unknown", "Fusion-PIC"),
  
  code_group("GAMESS", ["^GAMESS"], "unknown", "Quantum"),
  
  code_group("Gadget", ["GADGET", ".*-Gadget2", "L-PICOLA" ], "unknown", "N-Body"),
  
  code_group("gs2", [], "unknown", "unknown"),
  
  code_group("gtc", ["bench_gtc"], "unknown", "Fusion-PIC"),
  
  code_group("gts", ["^GTS"], "unknown", "Fusion-PIC"),
  
  code_group("GYRO", ["^BIGSCIENCE","gyro","tgyro", "cgyro"], "unknown", "Fusion-PIC"),
  
  code_group("K2PIPI", ["NOARCH\.V"], "unknown", "unknown"),
  
  code_group("LAMMPS", ["LMP"], "unknown", "MD"),
  
  code_group("MFDN", ["XMFDN"], "unknown", "Quantum"),
  
  code_group("M3D", [], "unknown", "Fusion-PIC"),
  
  #obviously, anybody might name their code "cori.x"
  #empirically, only chulwoo has done so, so it's milc
  #numactl could also be anyone, but 98% detar
  code_group("MILC",
             ["KS_SPECTRUM", "^ks_", "numactl","cori.x"],
             "enabled", "QCD"),
 
  code_group("NAMD", [], "unknown", "MD"),
  
  code_group("NWCHEM", [], "unknown", "Quantum"),
  
  code_group("mitgcm", [], "unknown", "Climate"),
  
  code_group("NCAR-LES", ["LESMPI"], "unknown", "Climate"),
  
  code_group("nimrod", [], "unknown", "Fusion-Cont."),
  
  code_group("Nyx", ["^NYX"], "unknown", "AMR"),
  
  #Omega3P is a parallel finite-element electrogmagnetic code 
  #for high-fidelity modeling of cavities.
  code_group("omega3p", [], "unknown", "Accel-PIC"),
  
  code_group("osiris", [], "unknown", "Fusion-PIC"),
  
  code_group("Overlap_Quark_Propagator", 
             ["^OVERLAP_INVERTER","^OVERLAP_DECOMP","^OVERLAP_CONNECT",
              "^OVERLAP_AXIAL","OVERLAP_2MP", "^overlap_roper", "overlap_collab",
              "overlap_contract", "overlap_exa", "overalp_ff", "overlap_glue", "overlap_bulk",
              "overlap_Meson", "overalp_Mom", "overlap_check_inver", "overlap_pion",
              "overlap_proton", "overlap_renorm", "ovlap_inverter"], 
             "unknown", "QCD"),
  
  code_group("parsec", [], "unknown", "DFT"),
  
  code_group("PEtot", ["PETOT"], "unknown", "DFT"),
  
  code_group("PIC", [], "unknown", "Fusion-PIC"),
  
  code_group("pop", [], "unknown", "Climate"),
  
  #http://iopscience.iop.org/0953-4075/39/17/017/pdf/0953-4075_39_17_017.pdf
  code_group("pstg", [], "unknown", "Quantum"),
  
  code_group("Python", ["^PYTHON"], "unknown", "Analytics"),
  
  code_group("qbox", [], "unknown", "DFT"),
  
  #not really amr, but uniform block structured grid
  code_group("S3D", ["^S3D"], "unknown", "AMR"),

  #Email from Ali Uzun
  #This is an in-house computational fluid dynamics (CFD) solver
  #We are performing simulations using as many as 24 billion grid points
  #The high Reynolds number necessitates a large number of grid points 
  #We changed the official name of this solver from SAURON to THORS
  #not really amr, but uniform block structured grid
  code_group("SAURON", ["SAURON"], "unknown", "AMR"),
  
  #desribed in 2011 NERSC NP workshop
  #radiation transport 
  #block structured or AMR
  code_group("Sedona3d", [], "unknown", "AMR"),
  
  code_group("su3", [], "unknown", "QCD"),
  
  code_group("WRF", ["^WRF.*"], "unknown", "Climate"),
  
  code_group("xgc", [], "unknown", "Fusion-PIC"),

  #columbia QCD
  code_group("CPS", ["main_multi_cpu"], "Proxy", "QCD"),

  code_group("Gromacs", ["mdrun", "rungromacs"], "unknown", "MD"),

  code_group("inv_k", ["inv_k"], "unknown", "unknown"),

  #arXiv:1806.07390v1 [astro-ph.IM] 19 Jun 2018
  code_group("Fornax", [], "unknown", "AMR"),

  code_group("ADDA", [], "unknown", "unknown"),

  #https://www.geosci-model-dev.net/9/927/2016/
  #bma guesses structured grid
  code_group("pflotran", [], "unknown", "AMR"),

  code_group("siesta", [], "unknown", "DFT"),

  code_group("aims", ["bup-aims"], "unknown", "unknown"),

  code_group("Pele", [], "unknown", "AMR"),

  code_group("AmrDerive", [], "unknown", "AMR"),

  code_group("main3d", [], "unknown", "unknown"),

  code_group("watershedDriver3d", [], "unknown", "AMR"),
  code_group("fractureTopeDriver3d", [], "unknown", "AMR"),
  code_group("advectTest", [], "unknown", "AMR"),

  code_group("BELLA", [], "unknown", "Accel-PIC"),

  code_group("nalu", [], "unknown", "unknown"),

  code_group("ACME", ["\d{4}-\d{2}-\d{2}\.csh"], "unknown", "Climate"),

  code_group("xlatqcd", [], "unknown", "QCD"),

  code_group("trans_er", [], "unknown", "unknown"),

  code_group("HACC", [], "Enabled", "N-Body"),

  code_group("Gaussian", ["g09"], "unknown", "Quantum"),

  code_group("BeamBeam3d", [], "unknown", "Accel-PIC"),

  code_group("Tristan", [], "unknown", "unknown"),

  code_group("amg2013", [], "unknown", "AMR"),

  code_group("GOOGLE", [], "unknown", "unknown"),

  code_group("launch", [], "unknown", "unknown"),

  code_group("osu_", [], "unknown", "Benchmarks & Profiling"),

  code_group("likwid", ["^\.likwidscript_"], "unknown", "Benchmarks & Profiling"),
  code_group("tfdist", ["^\.tfdist"], "unknown", "unknown" ),
  code_group("wraprun", ["^\.wraprun"], "unknown", "unknown"),

  code_group("totalview", ["cti_dlaunch1.0"], "unknown", "Debug"),

  code_group("JDBachan", ["\w{40}\.x"], "unknown", "Benchmarks & Profiling"),

  code_group("Pluto", [".*\.pluto\..*fuse.*\.exe",
                       "2mm\..*\.exe", "3mm\..*\.exe",
                       "adi\..*\.exe", "atax.*\.exe",
                       "bicg.*\.exe", "cholesky\..*\.exe",
                       "fdtd-", "floyd-warshall",
                       "gemm..*\.exe", "gemver..*\.exe", "gesummv\..*\.exe",
                       "jacobi-.d-imper.*\.exe",
                       "seidel2d.exe",
                       ".*c.chunked.c.icc.exe",
                       ".*c.chunked.ompsimd.c.icc.exe",
                       ".*c.icc.exe",
                       ".*chunked.c.icc.exe",
                       ".*chunked.ompsimd.c.icc.exe",
                       ".*ompsimd.c.icc.exe",
                       ".*chunked.c.icc.exe",
                       ".*chunked.ompsimd.c.icc.exe" ], 
             "unknown", "unknown"),

  code_group("Athena", ["^athena"], "unknown", "unknown"),

  code_group("BET", ["BET_.*\.exe"], "unknown", "unknown"),

  code_group("CHIMERA", [], "unknown", "unknown"),

  code_group("CHPL", [], "unknown", "unknown"),

  code_group("CNS3d", ["RNS\dd"], "unknown", "unknown"),

  code_group("CUTE", [], "unknown", "unknown"),

  code_group("ChaNGa", [], "unknown", "N-Body"),

  code_group("CoMD", [], "unknown", "MD"),

  code_group("ComWann", ["ComCoulomb", "ComDC", "ComLowH" ], "unknown", "unknown"),

  code_group("Compute-FiltDissipation", 
             ["Compute-Metrics", 
              "Compute-TurbDissipation"], 
             "unknown", "unknown"),

  code_group("DY_e", ["DY_e.*\.conf", "DY_mu.*\.conf" ], "unknown", "unknown"),

  code_group("Detect_cam5", 
             ["Detect_historical", "Detect_natural"
              "Extract_cam5", "Extract_happi", "Extract_historical", "Extract_natural", 
              "Map_happi", "Map_historical", "Map_natural" ], 
             "unknown", "Climate"),

  code_group("DiNu_RT", [], "unknown", "unknown"),

  code_group("EMPIRE_PIC", ["EPtran_driver"], "unknown", "unknown"),

  code_group("ESMC", ["ESMC_", "ESMF_"], "unknown", "Climate"),

  #http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html
  code_group("GIZMO", [], "unknown", "N-Body"),

  code_group("Maestro2d", [], "unknown", "AMR"),

  code_group("Maui3d", [], "unknown", "unknown"),

  code_group("PET_LAY", [], "unknown", "unknown"),

  code_group("U1Solver", [], "unknown", "unknown"),

  code_group("UAloop", ["UA_.*\.x"], "unknown", "unknown"),

  code_group("V_case", ["[IVX]*_case"], "unknown", "unknown"),

  code_group("a.out", [], "unknown", "unknown"),

  code_group("a-xxx", ["a-\d*x\d*x\d*x\d*-\d*-.*\.out"], "unknown", "unknown"),

  code_group("a_wait", ["a\d{10}_wait"], "unknown", "unknown"),

  code_group("a1", ["a5"], "unknown", "unknown"),

  code_group("abinit", [], "unknown", "unknown"),

  code_group("acc", [], "unknown", "unknown"),

  code_group("adios", [], "unknown", "Benchmarks & Profiling"),

  code_group("adj_gap_ratio", [], "unknown", "unknown"),

  code_group("afhyperneut", ["afneutron", "afnuc"], "unknown", "unknown"),

  code_group("barrier", [], "unknown", "Benchmarks & Profiling" ),
  code_group("atomic_tests", ["atomic"], "unknown", "unknown"),

  code_group("b.sh", ["b\d*\.sh"], "unknown", "unknown"),

  code_group("batch", [], "unknown", "unknown"),

  code_group("bb", ["bem-bb"], "unknown", "unknown"),

  code_group("hdf5", ["benchmark"], "unknown", "Benchmarks & Profiling"),

  #https://crd.lbl.gov/assets/Uploads/sc15-bigstick.pdf
  code_group("bigstick", [], "unknown", "Quantum"),

  code_group("bugs", ["bug\d*"], "unknown", "unknown"),

  code_group("Albany", [], "unknown", "unknown"),

  code_group("MPI_benchmarks", 
             ["AllReduce",
              "Allgather_test",
              "Allreduce_test",
              "Alltoall_test",
              "Alltoallv_test",
              "Bcast_test",
              "broadcast"
              "Broadcast_MPI",
              "Broadcast_Multi_MPIs",
              "Broadcast_Multi_OpenMP",
              "Broadcast_OpenMP",
              "IMB-", "IMPITester",
              "MPIVersion",
              "MPI_Allreduce",
              "MPI_Exec_r1",
              "MPI_PMRF", 
              "P2P_",
              "Scatter_",
              "Send_",
              "collect", "coll_perf",
              "iput", "isend_", "mpi",
              "overalp_test", "overlap_wins", "ping", 
              "randomMpiGet", "send_recv", "win_" ], 
             "unknown", "Benchmarks & Profiling"),

  code_group("UPC_benchmarks",
             [ "RandomAccess_UPC",
               "bench", "big_coarrray", "bigput", "bput",
               "bupc_", "fetch", "fma_", "get_", 
               "memcpy", "memory-leak", "memset", "pipelined", "put_",
               "sends.gasnet", "upc", "uts-" ], 
             "unknown", "Benchmarks & Profiling" ),

  code_group("NAS_benchmarsk",
             ["bt-mz\.","bt\.[ABCDE]", "cg", "dt.[ABCDSW]", 
              "ep-[ASW]", "ep.[ABCD]", "ep.out", "ep_server",
              "ft-[ASW]", "ft.[ABCD]",
              "is-[ASW]", "is.[ABCD]",
              "lu-b-bc", "lu-block", "lu\.[ABCD]",
              "mg\.[ABCD]",
              "sp-mz.C", "sp\.[ABCD]" ],
             "unknown", "Benchmarks & Profiling" ),

  code_group("Anderson", [], "unknown", "unknown"),
  
  code_group("CAT_", [], "unknown", "unknown"),

  code_group("CHM_", [], "unknown", "unknown"),

  code_group("CROSSECTION", [], "unknown", "unknown"),

  code_group("Calculate_PI", [], "unknown", "unknown"),

  code_group("Dense", [], "unknown", "unknown"),

  code_group("Petra", ["Epetra","Tpetra"], "unknown", "unknown"),

  code_group("Hei", [], "unknown", "unknown"),

  code_group("Impact", [], "unknown", "Accel-PIC"),

  code_group("JOB_n.sh", ["JOB_\d*\.sh"], "unknown", "unknown"),

  code_group("KernelRegression", [], "unknown", "unknown"),

  code_group("Kokkos", [], "unknown", "unknown"),

  #https://www.cray.com/blog/nerscs-edison-unleashed/
  #this links to the CCSE site - I think it's Maestro
  code_group("LMC3d", ["LMC2d", "LMCtest"], "unknown", "AMR"),

  code_group("MUSIC", [], "unknown", "unknown"),

  code_group("Maestro", [], "unknown", "AMR"),

  code_group("MolPro", [], "unknown", "Quantum"),

  code_group("Reorder", [], "unknown", "unknown"),

  code_group("SAM_ADV", [], "unknown", "unknown"),

  code_group("ScalarSolver", [], "unknown", "unknown"),

  code_group("ShengBTE", [], "unknown", "unknown"),

  #http://www.mcs.anl.gov/publication/simplemoc-performance-abstraction-3d-moc
    code_group("SimpleMOC", ["Simple"], "unknown", "unknown"),

  code_group("SpinTrack", [], "unknown", "unknown"),

  code_group("UMT", ["SuOlsonTest"], "unknown", "unknown"),

  code_group("VISIT", [], "unknown", "Analytics"),

  code_group("WaveSolver", [], "unknown", "unknown"),

  code_group("amrvis", [], "unknown", "AMR"),

  code_group("amr3d", ["amr.*\.ex"], "unknown", "AMR"),

  code_group("armci", [], "unknown", "Benchmarks & Profiling"),

  code_group("ascript", [], "unknown", "unknown"),

  code_group("asym", [], "unknown", "unknown"),

  code_group("async", [], "unknown", "unknown"),

  code_group("balloon", [], "unknown", "unknown"),

  code_group("bfs", [], "unknown", "unknown"),

  code_group("bipartite", [], "unknown", "unknown"),

  code_group("bundle.sh", ["bundle_\d*\.sh"], "unknown", "unknown"),

  code_group("Meraculous",
             [ "meraculous","contig","dumpK","genK",
               "kmer_", "noPref", 
               "overlap.*-KmerIndex", "genKmer",
               "overlap-noPref", "pkmer_", "testKmerIndex" ], 
             "unknown", "Analytics"),

  code_group("c11_", ["cxx_test"], "unknown", "unknown"),

  code_group("calcTau", [], "unknown", "unknown"),

  code_group("calc_decay", ["calc_.*_decay", "calc_auger", "calculate_"], "unknown", "unknown"),

  code_group("calcite3d", [], "unknown", "unknown"),

  code_group("cannon", [], "unknown", "unknown"),

  code_group("castep", [], "unknown", "Quantum"),

  code_group("check", [], "unknown", "unknown"),

  code_group("chiral_current", [], "unknown", "unknown"),

  code_group("SNAP", ["cksnap","ksmap","ksnap"], "unknown", "unknown"),

  code_group("clover", [], "unknown", "QCD"),

  code_group("cmc", [], "unknown", "unknown"),

  code_group("cmd.recon", [], "unknown", "unknown"),

  code_group("coarse_restrictor_profile", [], "unknown", "unknown"),

  code_group("cogent", [], "unknown", "unknown"),

  code_group("conftest", [], "unknown", "unknown"),

  code_group("cosmomc", [], "unknown", "Analytics"),

  code_group("crlcrl", [], "unknown", "unknown"),
  code_group("dft_", [], "unknown", "DFT"),
  code_group("dmft", [], "unknown", "Quantum"),
  code_group("e00.csh", ["e\d\d\..*\.csh", "e\d\d\.csh"], "unknown", "Climate"),
  code_group("elm_pb", [], "unknown", "unknown"),
  code_group("est_rates", ["est_..._rates_"], "unknown", "unknown"),
  code_group("BoxLib", 
             [".*Linux.64.CC.ftn.OPT.MPI.ex",
              ".*Linux.64.CC.ftn.DEBUG.MPI.ex",
              ".*Linux.64.CC.ftn.DEBUG.PETSC.ex",
              ".*Linux.64.CC.ftn.OPT.MPI.PETSC.ex",
              ".*Linux.CC.ftn.MPI.ex",
              ".*Linux.g\+\+.gfortran.MPI.OMP.ex",
              ".*Linux.64.CC.ftn.OPT.MPI.GNU.ex",
              ".*DODECANE_LU.ex",
              "convert3d",
              "hyperclaw3d", "perf_tests3d" ], 
             "unknown", "AMR"),

  code_group("ex", ["ex\d\d"], "unknown", "unknown"),
  code_group("exec", ["exec\d*"], "unknown", "unknown"),
  code_group("exec_dd", ["exec_\d*_\d"], "unknown", "unknown"),
  code_group("exec_ens", ["exec_ens.*\.csh"], "unknown", "Climate"),
  code_group("f90tst_", [], "unknown", "unknown"),
  code_group("fc_mpi32.*\.x", [], "unknown", "unknown"),

  code_group("flow_", [], "unknown", "unknown"),
  code_group("floyd_allpairs", [], "unknown", "unknown"),
  code_group("frg.sh", ["frg..\.sh"], "unknown", "unknown"),

  code_group("GlobalArrays", ["ga_.*\.x", "global_" ], 
             "unknown", "Benchmarks & Profiling"),

  code_group("fafqmc", [], "unknown", "unknown"),
  code_group("glue_", [], "unknown", "unknown"),
  code_group("gppKer", [], "unknown", "unknown"),
  code_group("gups", [], "unknown", "Benchmarks & Profiling"),
  code_group("gwl", [], "unknown", "unknown"),
  code_group("hdf5", ["h5"], "unknown", "Benchmarks & Profiling"),
  code_group("heatTransfer", [], "unknown", "unknown"),
  code_group("heisenberg", [], "unknown", "unknown"),
  code_group("hello", ["mpi-hello"], "unknown", "Benchmarks & Profiling"),
  code_group("himag", [], "unknown", "unknown"),
  code_group("hpgmg", [], "unknown", "Benchmarks & Profiling"),
  #what are all of these things with _st16 etc
  code_group("issue", ["issue\d\d"], "unknown", "unknown"),
  code_group("isx", [], "unknown", "unknown"),
  code_group("julia", [], "unknown", "Analytics"),
  code_group("kmeans", [], "unknown", "Analytics"),

  #this is LLNLs lassen mini-app implemented in charm
  code_group("lassen_charm", [], "unknown", "unknown"),

  code_group("lbs", [], "unknown", "unknown"),
  code_group("legacy", [], "unknown", "unknown"),
  code_group("lens2hat", [], "unknown", "unknown"),
  code_group("libpfasst", [], "unknown", "unknown"),
  code_group("linked_list", [], "unknown", "unknown"),
  code_group("lobpcg", ["runlobpcg"], "unknown", "unknown"),
  code_group("lock", [], "unknown", "unknown"),
  code_group("lulesh", [], "unknown", "unknown"),
  code_group("lve", [], "unknown", "unknown"),
  code_group("lya", [], "unknown", "unknown"),
  code_group("mapscript", [], "unknown", "unknown"),
  code_group("mat-", [], "unknown", "unknown"),
  code_group("mfix", [], "unknown", "unknown"),
  code_group("mkFit", [], "unknown", "unknown"),
  code_group("mkelly", [], "unknown", "unknown"),
  code_group("mobiliti", [], "unknown", "unknown"),
  code_group("mpmd.d.d-d", ["mpmd\.\d{8}\.\d{6}-\d{6}.conf"], "unknown", "unknown"),
  code_group("mpmd-d.conf", ["mpmd-\d\.conf"], "unknown", "unknown"),
  code_group("mpmd.conf", [], "unknown", "unknown"),
  code_group("mpmd.deepmd", [], "unknown", "unknown"),
  code_group("mpmd_hmm", [], "unknown", "unknown"),
  code_group("ocean_model", [], "unknown", "Climate"),

  code_group("omp_", [], "unknown", "Benchmarks & Profiling"),
  code_group("onetep", [], "unknown", "DFT"),

  code_group("operators_", [], "unknown", "unknown"),
  code_group("orca", [], "unknown", "unknown"),

  code_group("p1d_all", ["p1e\d", "p1eg\d", "p1eg\d\d" ], "unknown", "unknown"),
  code_group("pamd", [], "unknown", "MD"),

  code_group("parallel_fault", [], "unknown", "unknown"),
  code_group("paratec", [], "unknown", "DFT"),

  code_group("pddrive", ["pdtest"], "unknown", "unknown"),
  code_group("phold", [], "unknown", "unknown"),

  code_group("pimd", [], "unknown", "MD"),
  code_group("IO_benchmark", ["pio", "IOR", "netcdf", "pnetcdf"], 
             "unknown", "Benchmarks & Profiling"),

  code_group("plotRhoGrad", ["plotsoln"], "unknown", "unknown"),

  code_group("pom.conf", ["pom\d{23}\.conf"], "unknown", "unknown"),

  code_group("postw90", [], "unknown", "DFT"),

  code_group("pp_mpi", ["pp_u4", "pp_upcxx", "upcxx2"], 
             "unknown", "Benchmarks & Profiling"),

  code_group("ppp1", ["ppp2", "ppp_base" ], "unknown", "unknown"),
  code_group("pquadmc", [], "unknown", "unknown"),

  code_group("preqx", [], "unknown", "unknown"),
  code_group("pwdft", [], "unknown", "unknown"),

  code_group("toast", ["pyc_toast"], "unknown", "Analytics"),
  code_group("qchem", ["qcprog"], "unknown", "Quantum"),

  code_group("qlua", [], "unknown", "QCD"),
  code_group("qmcrunner", [], "unknown", "Quantum"),

  code_group("qparwcoll", [], "unknown", "unknown"),
  code_group("RemoteAtomics", 
             ["ra-.*-ugni-qthreads-",
              "reduction-.*-ugni-qthreads",
              "stencil-opt-.*-ugni-qthreads",
              "stream-.*-ugni-qthreads",
              "stream-ep-.*-ugni-qthreads"], 
             "unknown", "Benchmarks & Profiling"),

  code_group("rb1d", ["rb2d"], "unknown", "unknown"),
  code_group("rdma_performace", ["rdm_","rdma_"], "unknown", "Benchmarks & Profiling"),

  code_group("rhmc", [], "unknown", "QCD"),
  code_group("run-beam", [], "unknown", "unknown"),

  #binary star evolution
  #http://www.astro.wisc.edu/~bailey/thecode.html
  code_group("runFiltBSE", [], "unknown", "N-Body"),

  code_group("runMFKernel", [], "unknown", "unknown"),

  code_group("pselinv", ["run_pselinv"], "unknown", "unknown"),

  code_group("run_susy_train", [], "unknown", "unknown"),

  code_group("run_szpol", [], "unknown", "unknown"),

  code_group("sam.sh", ["sam.*\.sh"], "unknown", "unknown"),

  code_group("scf", [], "unknown", "Quantum"),

  code_group("sepsis", [], "unknown", "unknown"),

  code_group("sgw", [], "unknown", "unknown"),

  code_group("shiftcurrent", [], "unknown", "unknown"),

  code_group("shirley", [], "unknown", "unknown"),
  code_group("shmem", [], "unknown", "unknown"),

  code_group("simple", [], "unknown", "unknown"),
  code_group("sl_calc", ["sl_runLBLRTM", "sl_setup_LBLRTM" ], "unknown", "unknown"),

  code_group("small-ground", [], "unknown", "unknown"),
  code_group("spdyn", [], "unknown", "unknown"),

  code_group("spec.x", ["spec.*\.m"], "unknown", "unknown"),
  code_group("spectN", ["spectN\d*.*\.x"], "unknown", "unknown"),

  #not really amr, but block structured grids
  code_group("stencil3d", [], "unknown", "AMR"),
  code_group("stencil_", [], "unknown", "AMR"),
  code_group("stream3d", ["stream_budget3d"], "unknown", "AMR"),

  code_group("stream", [], "unknown", "Benchmarks & Profiling"),
  code_group("strided-bench", ["strided-comm", "strided_"], 
             "unknown", "Benchmarks & Profiling"),

  code_group("sw4", [], "unknown", "unknown"),
  code_group("swmr", [], "unknown", "unknown"),
  code_group("szscl21", [], "unknown", "unknown"),
  code_group("tbtrans", [], "unknown", "unknown"),
  code_group("tddft", [], "unknown", "DFT"),
  code_group("test", ["tst_"], "unknown", "unknown"),
  code_group("try_", [], "unknown", "unknown"),
  code_group("v.flow", [], "unknown", "unknown"),
  code_group("vpic", [], "unknown", "Accel-PIC"),
  code_group("vorpal", [], "unknown", "Accel-PIC"),
  code_group("wannier90", [], "unknown", "DFT"),
  code_group("warp", [], "unknown", "unknown"),
  code_group("weh2d", [], "unknown", "unknown"),
  code_group("whetstone", [], "unknown", "Benchmarks & Profiling"),
  code_group("write", [], "unknown", "unknown"),
  code_group("WannierTools", ["wt.x", "wt-.*\.x", "wt_.*\.x"], "unknown", "DFT"),
  code_group("xfastran", [], "unknown", "unknown"),
  code_group("xemir", [], "unknown", "unknown"),
  code_group("xgenray", [], "unknown", "unknown"),
  code_group("xlatqcd", [], "unknown", "QCD"),
  code_group("xmain", [], "unknown", "unknown"),
  code_group("xi_", [], "unknown", "unknown"),
  code_group("xmerge", [], "unknown", "unknown"),
  code_group("xnet", [], "unknown", "unknown"),
  code_group("xparvmec", [], "unknown", "unknown"),
  code_group("xsolver", [], "unknown", "unknown"),
  code_group("xthi", [], "unknown", "Benchmarks & Profiling"),
  code_group("z_ptest", [], "unknown", "unknown"),
  #kinetic monte carlo
  code_group("zacros", [], "unknown", "unknown"),
  code_group("6f_landau", [], "unknown", "unknown"),
  code_group("AFDMC", [], "unknown", "unknown"),
  code_group("ALaDyn", [], "unknown", "unknown"),
  code_group("Crosssection", [], "unknown", "unknown"),
  code_group("NPmpi", [], "unknown", "unknown"),
  code_group("Sbm_", [], "unknown", "unknown"),
  code_group("SuNF", [], "unknown", "unknown"),
  code_group("TopOpt", [], "unknown", "unknown"),
  code_group("desi", [], "unknown", "Analytics"),
  #code_group("", [], "unknown", "unknown"),
  #code_group("", [], "unknown", "unknown"),
  #code_group("", [], "unknown", "unknown"),

]#closes CODE_GROUPS

CODE_GROUPS_D = { r:g for g in CODE_GROUPS for r in g.pattern }




