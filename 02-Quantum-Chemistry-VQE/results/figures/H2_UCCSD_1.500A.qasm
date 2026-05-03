OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(12.132832762382435) q0; }
gate gate_PauliEvolution_2877389523216(param0) q0,q1 { ry(6.716723076449403) q1; }
gate gate_PauliEvolution_2877389521776(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(0.004775864798654524) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-0.004775864798654524) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(6.066416381191219) q0,q1; gate_PauliEvolution_2877389523216(-3.358361538224703) q0,q1; gate_PauliEvolution_2877389521776(-0.004775864798654526) q0,q1; }
qreg q[2];
gate_EvolvedOps(6.066416381191219,-3.358361538224703,-0.004775864798654526) q[0],q[1];