// Top module that instantiates sub_counter from a separate file
module top_multi (
    input  logic        clk,
    input  logic        rst,
    output logic [15:0] result
);
    logic [7:0] count_a, count_b;

    sub_counter #(.WIDTH(8)) u_a (
        .clk(clk), .rst(rst), .count(count_a)
    );

    sub_counter #(.WIDTH(8)) u_b (
        .clk(clk), .rst(rst), .count(count_b)
    );

    assign result = {count_a, count_b};
endmodule
