// Test generate-for: chain of 4 adders with parameterized width
module generate_chain (
    input  logic        clk,
    input  logic        rst,
    output logic [7:0]  result
);
    localparam N = 4;

    // LFSR for stimulus
    logic [7:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 8'hA5;
        else
            lfsr <= {lfsr[6:0], lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]};
    end

    // Chain of additions: stage[0] = lfsr, stage[i+1] = stage[i] + i
    logic [7:0] stage [0:N];
    assign stage[0] = lfsr;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_add
            assign stage[i+1] = stage[i] + 8'(i + 1);
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst)
            result <= 8'd0;
        else
            result <= stage[N];
    end
endmodule
