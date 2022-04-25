class Vehicle:
    def __init__(self, color, price, type):
        self.color = color 
        self.price = price
        self.type = type
    def drive(self):
        print (self.type, self.color, 'поехал')    
    def stop(self):
        print(self.type, self.color'остановился')

car1 = Vehicle("Красный", 49999, "Легковой")
print(car1.color, car1.price, car1.type)
car1.drive()
car1.stop()