OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-15.278566467433668) q0; }
gate gate_PauliEvolution_2877353323856(param0) q0,q1 { ry(15.277945420124125) q1; }
gate gate_PauliEvolution_2877353316336(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(1.4800980114441267) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-1.4800980114441267) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-7.639283233716837) q0,q1; gate_PauliEvolution_2877353323856(-7.638972710062065) q0,q1; gate_PauliEvolution_2877353316336(-1.4800980114441271) q0,q1; }
qreg q[2];
gate_EvolvedOps(-7.639283233716837,-7.638972710062065,-1.4800980114441271) q[0],q[1];