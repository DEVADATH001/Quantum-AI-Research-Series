OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(0.4367993753087126) q0; }
gate gate_PauliEvolution_1551705586608(param0) q0,q1 { ry(-0.43874580329129226) q1; }
gate gate_PauliEvolution_1551705586768(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(3.1469190490414443) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-3.1469190490414443) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(0.21839968765435638) q0,q1; gate_PauliEvolution_1551705586608(0.2193729016456462) q0,q1; gate_PauliEvolution_1551705586768(-3.146919049041445) q0,q1; }
qreg q[2];
gate_EvolvedOps(0.21839968765435638,0.2193729016456462,-3.146919049041445) q[0],q[1];