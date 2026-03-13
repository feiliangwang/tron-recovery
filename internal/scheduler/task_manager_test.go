package scheduler

import (
	"testing"
)

func newTestManager(t *testing.T) *TaskManager {
	t.Helper()
	tm, err := NewTaskManager(t.TempDir())
	if err != nil {
		t.Fatalf("NewTaskManager failed: %v", err)
	}
	return tm
}

func TestNewTaskManager(t *testing.T) {
	tm := newTestManager(t)
	if tm == nil {
		t.Fatal("NewTaskManager returned nil")
	}
	if len(tm.ListJobs()) != 0 {
		t.Error("new TaskManager should have no jobs")
	}
}

func TestCreateJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test job", "word1 ? word3", 1000)

	if job == nil {
		t.Fatal("CreateJob returned nil")
	}
	if job.ID == "" {
		t.Error("job ID should not be empty")
	}
	if job.Name != "test job" {
		t.Errorf("Name = %q, want %q", job.Name, "test job")
	}
	if job.Mnemonic != "word1 ? word3" {
		t.Errorf("Mnemonic = %q, want %q", job.Mnemonic, "word1 ? word3")
	}
	if job.BatchSize != 1000 {
		t.Errorf("BatchSize = %d, want 1000", job.BatchSize)
	}
	if job.Status != "pending" {
		t.Errorf("Status = %q, want %q", job.Status, "pending")
	}
	if job.CreatedAt.IsZero() {
		t.Error("CreatedAt should not be zero")
	}
}

func TestCreateJobAutoID(t *testing.T) {
	tm := newTestManager(t)
	j1 := tm.CreateJob("job 1", "mnemonic1", 100)
	j2 := tm.CreateJob("job 2", "mnemonic2", 100)
	if j1.ID == j2.ID {
		t.Error("two jobs should have different IDs")
	}
}

func TestGetJob(t *testing.T) {
	tm := newTestManager(t)
	created := tm.CreateJob("test", "mnemonic", 100)

	got := tm.GetJob(created.ID)
	if got == nil {
		t.Fatal("GetJob returned nil for existing job")
	}
	if got.ID != created.ID {
		t.Errorf("GetJob returned wrong job: ID = %q, want %q", got.ID, created.ID)
	}
}

func TestGetJobNotFound(t *testing.T) {
	tm := newTestManager(t)
	got := tm.GetJob("job-999")
	if got != nil {
		t.Error("GetJob should return nil for non-existent job")
	}
}

func TestListJobs(t *testing.T) {
	tm := newTestManager(t)
	tm.CreateJob("job 1", "mnemonic1", 100)
	tm.CreateJob("job 2", "mnemonic2", 100)
	tm.CreateJob("job 3", "mnemonic3", 100)

	jobs := tm.ListJobs()
	if len(jobs) != 3 {
		t.Errorf("ListJobs() count = %d, want 3", len(jobs))
	}
}

func TestUpdateJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("original", "mnemonic", 100)
	job.Name = "updated"
	tm.UpdateJob(job)

	got := tm.GetJob(job.ID)
	if got.Name != "updated" {
		t.Errorf("Name after update = %q, want %q", got.Name, "updated")
	}
}

func TestStartJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)

	ok := tm.StartJob(job.ID)
	if !ok {
		t.Error("StartJob should return true for pending job")
	}

	got := tm.GetJob(job.ID)
	if got.Status != "running" {
		t.Errorf("Status after Start = %q, want %q", got.Status, "running")
	}
	if got.StartedAt == nil {
		t.Error("StartedAt should be set after Start")
	}
}

func TestStartJobNotFound(t *testing.T) {
	tm := newTestManager(t)
	ok := tm.StartJob("job-999")
	if ok {
		t.Error("StartJob should return false for non-existent job")
	}
}

func TestStartJobCompleted(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	tm.StartJob(job.ID)
	tm.CompleteJob(job.ID)

	ok := tm.StartJob(job.ID)
	if ok {
		t.Error("StartJob should return false for completed job")
	}
}

func TestPauseJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	tm.StartJob(job.ID)

	ok := tm.PauseJob(job.ID)
	if !ok {
		t.Error("PauseJob should return true for running job")
	}

	got := tm.GetJob(job.ID)
	if got.Status != "paused" {
		t.Errorf("Status after Pause = %q, want %q", got.Status, "paused")
	}
}

func TestPauseJobNotRunning(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	// job is "pending", not "running"
	ok := tm.PauseJob(job.ID)
	if ok {
		t.Error("PauseJob should return false for non-running job")
	}
}

func TestResumeJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	tm.StartJob(job.ID)
	tm.PauseJob(job.ID)

	ok := tm.ResumeJob(job.ID)
	if !ok {
		t.Error("ResumeJob should return true for paused job")
	}

	got := tm.GetJob(job.ID)
	if got.Status != "running" {
		t.Errorf("Status after Resume = %q, want %q", got.Status, "running")
	}
}

func TestResumeJobNotPaused(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	// job is "pending", not "paused"
	ok := tm.ResumeJob(job.ID)
	if ok {
		t.Error("ResumeJob should return false for non-paused job")
	}
}

func TestDeleteJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)

	ok := tm.DeleteJob(job.ID)
	if !ok {
		t.Error("DeleteJob should return true for existing job")
	}

	if tm.GetJob(job.ID) != nil {
		t.Error("GetJob should return nil after deletion")
	}
	if len(tm.ListJobs()) != 0 {
		t.Error("ListJobs should be empty after deleting the only job")
	}
}

func TestDeleteJobNotFound(t *testing.T) {
	tm := newTestManager(t)
	ok := tm.DeleteJob("job-999")
	if ok {
		t.Error("DeleteJob should return false for non-existent job")
	}
}

func TestCompleteJob(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)
	tm.StartJob(job.ID)
	tm.CompleteJob(job.ID)

	got := tm.GetJob(job.ID)
	if got.Status != "completed" {
		t.Errorf("Status after Complete = %q, want %q", got.Status, "completed")
	}
	if got.CompletedAt == nil {
		t.Error("CompletedAt should be set after Complete")
	}
}

func TestIncrementCompleted(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)

	tm.IncrementCompleted(job.ID, 5)
	tm.IncrementCompleted(job.ID, 3)

	got := tm.GetJob(job.ID)
	if got.Completed != 2 {
		t.Errorf("Completed = %d, want 2", got.Completed)
	}
	if got.Matches != 8 {
		t.Errorf("Matches = %d, want 8", got.Matches)
	}
}

func TestSetTotal(t *testing.T) {
	tm := newTestManager(t)
	job := tm.CreateJob("test", "mnemonic", 100)

	tm.SetTotal(job.ID, 4194304)

	got := tm.GetJob(job.ID)
	if got.Total != 4194304 {
		t.Errorf("Total = %d, want 4194304", got.Total)
	}
}

func TestStatePersistence(t *testing.T) {
	dir := t.TempDir()

	// Create manager and add jobs
	tm1, err := NewTaskManager(dir)
	if err != nil {
		t.Fatalf("NewTaskManager failed: %v", err)
	}
	j1 := tm1.CreateJob("persistent job", "some mnemonic", 500)
	tm1.StartJob(j1.ID)
	tm1.SetTotal(j1.ID, 999)

	// Create a new manager from the same directory
	tm2, err := NewTaskManager(dir)
	if err != nil {
		t.Fatalf("NewTaskManager reload failed: %v", err)
	}

	jobs := tm2.ListJobs()
	if len(jobs) != 1 {
		t.Fatalf("after reload, job count = %d, want 1", len(jobs))
	}

	got := tm2.GetJob(j1.ID)
	if got == nil {
		t.Fatal("job not found after reload")
	}
	if got.Name != "persistent job" {
		t.Errorf("Name = %q, want %q", got.Name, "persistent job")
	}
	if got.Total != 999 {
		t.Errorf("Total = %d, want 999", got.Total)
	}
}

func TestAutoIDRestoredOnReload(t *testing.T) {
	dir := t.TempDir()

	tm1, _ := NewTaskManager(dir)
	j1 := tm1.CreateJob("job1", "m1", 100)
	j2 := tm1.CreateJob("job2", "m2", 100)
	_ = j2

	// New manager should continue IDs from where tm1 left off
	tm2, _ := NewTaskManager(dir)
	j3 := tm2.CreateJob("job3", "m3", 100)

	if j3.ID == j1.ID {
		t.Error("new job should not have the same ID as previous job after reload")
	}
}
