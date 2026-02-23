// Parameterized adder chain to test generate blocks and parameter overrides
module adder_unit #(
    parameter WIDTH = 8
)(
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] sum
);
    assign sum = a + b;
endmodule

module param_adder (
    input  logic        clk,
    input  logic        rst,
    output logic [15:0] result
);
    // LFSR for stimulus
    logic [15:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

    // Instantiate 16-bit adder with parameter override
    logic [15:0] sum_wide;
    adder_unit #(.WIDTH(16)) u_add (
        .a(lfsr),
        .b(result),
        .sum(sum_wide)
    );

    // Accumulate
    always_ff @(posedge clk) begin
        if (rst)
            result <= 16'd0;
        else
            result <= sum_wide;
    end
endmodule
