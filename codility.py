def solution(n):
    gates = []

    def add_multiplex(gate, target, control):

        for i in range(2 ** len(control)):
            gates.append(gate + " " + str(target) + "\n")

            next_i = i + 1
            if next_i == 2 ** len(control):
                next_i = 0

            first_gray = i ^ (i >> 1)
            second_gray = next_i ^ (next_i >> 1)

            different = first_gray ^ second_gray
            bit = 0

            while different > 1:
                different = different // 2
                bit = bit + 1

            control = control[bit]

            gates.append("cnot " + str(target) + " " + str(control) + "\n")

    def qsd(qubits):
        if len(qubits) == 1:
            qubit = qubits[0]

            gates.append("rz " + str(qubit) + "\n")
            gates.append("ry " + str(qubit) + "\n")
            gates.append("rz " + str(qubit) + "\n")

            return

        target = qubits[0]
        controls = qubits[1:]

        qsd(controls)
        add_multiplex("rz", target, controls)

        qsd(controls)
        add_multiplex("ry", target, controls)

        qsd(controls)
        add_multiplex("rz", target, controls)

        qsd(controls)

    qubits = []

    for i in range(n):
        qubits.append(i)

    qsd(qubits)

    return "".join(gates)