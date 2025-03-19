# Student class definition
class Student:
    def __init__(self, student_id, name, age):
        self.student_id = student_id
        self.name = name
        self.age = age

    def display_info(self):
        print(f"학번: {self.student_id}, 이름: {self.name}, 나이: {self.age}")


# StudentManager class definition
class StudentManager:
    def __init__(self):
        self.students = []  # List to store students

    def add_student(self, student):
        self.students.append(student)   # Adds a student to the list

    def display_all_students(self):
        for student in self.students:
            student.display_info()  #Calls display_info for each student


# Creating instances of Student

student1 = Student("1번", "김철수", "20살")
student2 = Student("2번", "이영희", "21살")
student3 = Student("3번", "박지민", "19살")

# Creating an instance of StudentManager and adding the first 3 students
manager = StudentManager()
manager.add_student(student1)
manager.add_student(student2)
manager.add_student(student3)

# Display the information of the first 3 students
print("현재 등록된 학생 목록")
manager.display_all_students()

# Adding a 4th student
student4 = Student("4번", "한진수", "22살")
print("\n4번 학생 추가 후")
manager.add_student(student4)

# Display the information of all 4 students
manager.display_all_students()
