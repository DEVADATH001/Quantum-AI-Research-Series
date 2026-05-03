OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(12.131149590693234) q0; }
gate gate_PauliEvolution_1913336674608(param0) q0,q1 { ry(-5.8479772081553465) q1; }
gate gate_PauliEvolution_1913336675088(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(0.0048404490818217145) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-0.0048404490818217145) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(6.065574795346619) q0,q1; gate_PauliEvolution_1913336674608(2.923988604077674) q0,q1; gate_PauliEvolution_1913336675088(-0.004840449081821716) q0,q1; }
qreg q[2];
gate_EvolvedOps(6.065574795346619,2.923988604077674,-0.004840449081821716) q[0],q[1];