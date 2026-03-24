// Testbench automatico — nand_add2
// 16 vetores de teste

`timescale 1ns/1ps

module nand_add2_tb;

    reg  x0, x1, x2, x3;
    wire y0, y1, y2;

    nand_add2 dut (.x0(x0), .x1(x1), .x2(x2), .x3(x3), .y0(y0), .y1(y1), .y2(y2));

    integer errors = 0;

    initial begin
        $display("Testbench: nand_add2");
        $display("----------------------------------------");

        x0 = 0; x1 = 0; x2 = 0; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b000) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 0, 0, 0, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 0, 0, 0);
        end

        x0 = 0; x1 = 0; x2 = 1; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b001) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 0, 1, 1, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 0, 1, 1);
        end

        x0 = 0; x1 = 0; x2 = 0; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b010) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 0, 2, 2, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 0, 2, 2);
        end

        x0 = 0; x1 = 0; x2 = 1; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b011) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 0, 3, 3, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 0, 3, 3);
        end

        x0 = 1; x1 = 0; x2 = 0; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b001) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 1, 0, 1, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 1, 0, 1);
        end

        x0 = 1; x1 = 0; x2 = 1; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b010) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 1, 1, 2, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 1, 1, 2);
        end

        x0 = 1; x1 = 0; x2 = 0; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b011) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 1, 2, 3, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 1, 2, 3);
        end

        x0 = 1; x1 = 0; x2 = 1; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b100) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 1, 3, 4, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 1, 3, 4);
        end

        x0 = 0; x1 = 1; x2 = 0; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b010) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 2, 0, 2, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 2, 0, 2);
        end

        x0 = 0; x1 = 1; x2 = 1; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b011) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 2, 1, 3, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 2, 1, 3);
        end

        x0 = 0; x1 = 1; x2 = 0; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b100) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 2, 2, 4, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 2, 2, 4);
        end

        x0 = 0; x1 = 1; x2 = 1; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b101) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 2, 3, 5, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 2, 3, 5);
        end

        x0 = 1; x1 = 1; x2 = 0; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b011) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 3, 0, 3, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 3, 0, 3);
        end

        x0 = 1; x1 = 1; x2 = 1; x3 = 0; #10;
        if ({y2, y1, y0} !== 3'b100) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 3, 1, 4, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 3, 1, 4);
        end

        x0 = 1; x1 = 1; x2 = 0; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b101) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 3, 2, 5, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 3, 2, 5);
        end

        x0 = 1; x1 = 1; x2 = 1; x3 = 1; #10;
        if ({y2, y1, y0} !== 3'b110) begin
            $display("FAIL: add(%0d,%0d)=%0d got=%0d", 3, 3, 6, {y2, y1, y0});
            errors = errors + 1;
        end else begin
            $display("PASS: add(%0d,%0d)=%0d", 3, 3, 6);
        end

        $display("----------------------------------------");
        $display("Erros: %0d / 16", errors);
        if (errors == 0) $display("CIRCUITO CORRETO");
        else             $display("CIRCUITO COM FALHAS");
        $finish;
    end

endmodule