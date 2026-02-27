interface simple_if;
    logic [7:0] data;
    logic       valid;
    modport src(output data, valid);
    modport dst(input data, valid);
endinterface

module writer(simple_if.src bus, input logic clk, input logic rst);
    always_ff @(posedge clk) begin
        if (rst) begin
            bus.data <= 8'd0;
            bus.valid <= 1'b0;
        end else begin
            bus.data <= bus.data + 8'd1;
            bus.valid <= 1'b1;
        end
    end
endmodule

module reader(simple_if.dst bus, input logic clk, input logic rst, output logic [31:0] result);
    logic [31:0] checksum;
    always_ff @(posedge clk) begin
        if (rst) begin
            checksum <= 32'd0;
            result <= 32'd0;
        end else begin
            if (bus.valid)
                checksum <= checksum ^ {24'd0, bus.data};
            result <= checksum;
        end
    end
endmodule

module iface_modport(input logic clk, input logic rst, output logic [31:0] result);
    simple_if bus();
    writer w(.bus(bus), .clk(clk), .rst(rst));
    reader r(.bus(bus), .clk(clk), .rst(rst), .result(result));
endmodule
