// Gerado por NANDNet
// Operador    : add  n_bits : 2
// Portas raw  : 17
// Apos poda   : 4 (13 mortas removidas)
// gamma_crisp : 20.0

module nand_add2 (
    input  wire x0, x1, x2, x3,
    output wire y0, y1, y2
);

    wire l1_n1, l2_n1, l4_n2;

    assign y0 = l2_n1;  // deduplicado
    assign y1 = l2_n1;  // deduplicado

    // camada 1
    nand g_1_1 (l1_n1, x1, x3);
    // camada 2
    nand g_2_1 (l2_n1, l1_n1, l1_n1);  // NOT(l1_n1)
    // camada 4
    nand g_4_2 (l4_n2, x3, x0);
    // saida
    nand g_5_2 (y2, l4_n2, l1_n1);

endmodule