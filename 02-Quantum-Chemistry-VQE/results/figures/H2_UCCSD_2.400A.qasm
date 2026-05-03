OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(0.43009783987847394) q0; }
gate gate_PauliEvolution_2877352802608(param0) q0,q1 { ry(12.136818363825057) q1; }
gate gate_PauliEvolution_2877352804048(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(3.145930579873461) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-3.145930579873461) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(0.21504891993923705) q0,q1; gate_PauliEvolution_2877352802608(-6.06840918191253) q0,q1; gate_PauliEvolution_2877352804048(-3.145930579873462) q0,q1; }
qreg q[2];
gate_EvolvedOps(0.21504891993923705,-6.06840918191253,-3.145930579873462) q[0],q[1];