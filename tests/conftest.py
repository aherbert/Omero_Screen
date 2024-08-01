import pytest
from typing import List, Dict, Any, Optional

class MockOMEROObject:
    def __init__(self, id: int, name: str, obj_type: str):
        self.id = id
        self.name = name
        self.obj_type = obj_type
        self._annotations = []
        self._children = []

    def getName(self) -> str:
        return self.name

    def getId(self) -> int:
        return self.id

    def listAnnotations(self) -> List['MockAnnotation']:
        return self._annotations

    def addAnnotation(self, annotation: 'MockAnnotation') -> None:
        self._annotations.append(annotation)

    def getAnnotation(self, namespace: str) -> Optional['MockAnnotation']:
        return next((ann for ann in self._annotations if ann.getNs() == namespace), None)

    def listChildren(self) -> List['MockOMEROObject']:
        return self._children

    def addChild(self, child: 'MockOMEROObject') -> None:
        self._children.append(child)

class MockAnnotation:
    def __init__(self, namespace: str, value: Any):
        self.namespace = namespace
        self.value = value

    def getNs(self) -> str:
        return self.namespace

    def getValue(self) -> Any:
        return self.value

class MockMapAnnotation(MockAnnotation):
    def __init__(self, namespace: str, value: List[List[str]]):
        super().__init__(namespace, value)

    def setValue(self, value: List[List[str]]) -> None:
        self.value = value

    def save(self) -> None:
        pass  # In a real scenario, this would save the annotation

class MockFileAnnotation(MockAnnotation):
    def __init__(self, namespace: str, file_name: str, file_content: bytes):
        super().__init__(namespace, None)
        self.file_name = file_name
        self.file_content = file_content

    def getFile(self) -> 'MockOriginalFile':
        return MockOriginalFile(self.file_name, self.file_content)

class MockOriginalFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content

    def getName(self) -> str:
        return self.name

    def asFileObj(self) -> List[bytes]:
        return [self.content]

class MockBlitzGateway:
    def __init__(self):
        self.objects = {}

    def getObject(self, obj_type: str, obj_id: int) -> Optional[MockOMEROObject]:
        return self.objects.get((obj_type, obj_id))

    def getObjects(self, obj_type: str, opts: Dict[str, Any], attributes: Dict[str, Any]) -> List[MockOMEROObject]:
        return [obj for obj in self.objects.values() if obj.obj_type == obj_type and all(getattr(obj, k, None) == v for k, v in attributes.items())]

    def addObject(self, obj: MockOMEROObject) -> None:
        self.objects[(obj.obj_type, obj.id)] = obj

    def getUser(self) -> MockOMEROObject:
        return MockOMEROObject(1, "Test User", "Experimenter")

    def getUpdateService(self) -> 'MockUpdateService':
        return MockUpdateService()

class MockUpdateService:
    def saveObject(self, obj: Any) -> None:
        pass  # In a real scenario, this would save the object

@pytest.fixture
def mock_conn():
    return MockBlitzGateway()

@pytest.fixture
def mock_plate(mock_conn):
    plate = MockOMEROObject(1, "Test Plate", "Plate")
    mock_conn.addObject(plate)
    return plate

@pytest.fixture
def mock_project(mock_conn):
    project = MockOMEROObject(1, "Screens", "Project")
    mock_conn.addObject(project)
    return project

@pytest.fixture
def mock_dataset(mock_conn, mock_project):
    dataset = MockOMEROObject(1, "Test Dataset", "Dataset")
    mock_conn.addObject(dataset)
    mock_project.addChild(dataset)
    return dataset

# Add more fixtures as needed for your specific tests