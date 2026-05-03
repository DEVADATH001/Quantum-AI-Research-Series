OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-2.7025886836365296) q0; }
gate gate_PauliEvolution_2029344557168(param0) q0,q1 { ry(-3.5800558562921387) q1; }
gate gate_PauliEvolution_2029344557328(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(-4.806739142710384) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(4.806739142710384) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-1.3512943418182652) q0,q1; gate_PauliEvolution_2029344557168(1.79002792814607) q0,q1; gate_PauliEvolution_2029344557328(4.806739142710386) q0,q1; }
qreg q[2];
gate_EvolvedOps(-1.3512943418182652,1.79002792814607,4.806739142710386) q[0],q[1];