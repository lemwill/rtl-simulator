interface simple_bus;
    logic [7:0] data;
    logic       valid;
    logic       ready;
endinterface

module producer(simple_bus bus, input logic clk, input logic rst);
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

module consumer(simple_bus bus, input logic clk, input logic rst, output logic [31:0] result);
    logic [31:0] checksum;
    always_ff @(posedge clk) begin
        if (rst) begin
            checksum <= 32'd0;
            result <= 32'd0;
            bus.ready <= 1'b0;
        end else begin
            bus.ready <= 1'b1;
            if (bus.valid && bus.ready)
                checksum <= checksum ^ {24'd0, bus.data};
            result <= checksum;
        end
    end
endmodule

module iface_bus(input logic clk, input logic rst, output logic [31:0] result);
    simple_bus bus();
    producer p(.bus(bus), .clk(clk), .rst(rst));
    consumer c(.bus(bus), .clk(clk), .rst(rst), .result(result));
endmodule
