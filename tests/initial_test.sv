// Test: initial blocks, assignment patterns
module initial_test(
    input  logic        clk, rst,
    output logic [31:0] result
);
    logic [31:0] counter;
    logic [7:0]  magic;
    logic [7:0]  mem [0:3];

    // Initial block: set scalar and array elements
    initial begin
        magic = 8'hAB;
        mem[0] = 8'h10;
        mem[1] = 8'h20;
        mem[2] = 8'h30;
        mem[3] = 8'h40;
    end

    always_ff @(posedge clk) begin
        if (rst)
            counter <= 32'd0;
        else begin
            counter <= counter + 32'd1;
            mem[0] <= mem[0] + 8'd1;
        end
    end

    assign result = {mem[3], mem[2], mem[1], mem[0]}
                   + counter
                   + {24'd0, magic};
endmodule
