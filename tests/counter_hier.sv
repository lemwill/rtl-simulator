// Multi-module hierarchy test: two independent counters
module inner_counter (
    input  logic       clk,
    input  logic       rst,
    output logic [7:0] count
);
    always_ff @(posedge clk) begin
        if (rst)
            count <= 8'd0;
        else
            count <= count + 8'd1;
    end
endmodule

module counter_hier (
    input  logic       clk,
    input  logic       rst,
    output logic [7:0] count_a,
    output logic [7:0] count_b
);
    inner_counter u_a (.clk(clk), .rst(rst), .count(count_a));
    inner_counter u_b (.clk(clk), .rst(rst), .count(count_b));
endmodule
