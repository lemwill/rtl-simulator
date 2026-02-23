module struct_test(
    input logic clk, rst,
    output logic [31:0] result
);
    typedef struct packed {
        logic [15:0] x;
        logic [15:0] y;
    } point_t;

    point_t pt;

    always_ff @(posedge clk) begin
        if (rst) begin
            pt.x <= 16'd0;
            pt.y <= 16'd0;
        end else begin
            pt.x <= pt.x + 16'd1;
            pt.y <= pt.y + 16'd3;
        end
    end

    assign result = pt;
endmodule
