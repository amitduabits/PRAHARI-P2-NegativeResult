"""Reference implementations and experiment harnesses for the PRAHARI paper programme.

Each subpackage is one paper from `Research papers/PRAHARI_Research_Strategy.md`:

    p1_provenance  Paper 1  Provenance-aware inference dispatch          (CVPR)
    p2_fallback    Paper 2  Deterministic fallback engines               (ICCV)
    p3_nextcam     Paper 3  Implicit motion models / next-camera         (IJCAI)
    p4_admission   Paper 4  Deterministic concurrent decoder management  (TMM)
    p5_fusion      Paper 5  Cross-modal collapse and alert dedup         (TCSVT)
    p6_platform    Paper 6  Multi-authority platform design              (TETC)

The implementations mirror the semantics of the shipped system under
02_Code/prahari (app/services/*). Each module names the production module it
mirrors so a reviewer can check that the paper describes the deployed system.
"""

__version__ = "0.1.0"
