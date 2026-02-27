module casez_decode(input logic clk, input logic rst, output logic [31:0] result);
    logic [3:0] cnt;
    logic [31:0] acc;
    logic [7:0] decoded;

    // Combinational decode with casez wildcards
    always_comb begin
        casez (cnt)
            4'b1???: decoded = 8'd10;  // 8-15: MSB set
            4'b01??: decoded = 8'd20;  // 4-7
            4'b001?: decoded = 8'd30;  // 2-3
            4'b0001: decoded = 8'd40;  // 1
            4'b0000: decoded = 8'd50;  // 0
            default: decoded = 8'd0;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            cnt <= 4'd0;
            acc <= 32'd0;
            result <= 32'd0;
        end else begin
            cnt <= cnt + 4'd1;
            acc <= acc + {24'd0, decoded};
            result <= acc;
        end
    end
endmodule
