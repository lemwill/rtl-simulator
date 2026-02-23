module lfsr (
    input  logic       clk,
    input  logic       rst,
    output logic [7:0] lfsr_out
);
    // Galois LFSR: x^8 + x^6 + x^5 + x^4 + 1
    // Taps at bits 7, 5, 4, 3 (0-indexed)
    always_ff @(posedge clk) begin
        if (rst)
            lfsr_out <= 8'd1;
        else
            lfsr_out <= {lfsr_out[6:0], lfsr_out[7] ^ lfsr_out[5] ^ lfsr_out[4] ^ lfsr_out[3]};
    end
endmodule
