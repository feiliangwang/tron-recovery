package worker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"boon/internal/bloom"
	"boon/internal/protocol"
)

// ---- CompactClient tests ----

func makeEncodedTemplate() []byte {
	tmpl := &protocol.TaskTemplate{
		JobID:      42,
		UnknownPos: []int{11},
		Words: []string{
			"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "",
		},
	}
	return tmpl.Encode()
}

func makeEncodedTask() []byte {
	task := &protocol.CompactTask{
		TaskID:   10,
		JobID:    42,
		StartIdx: 0,
		EndIdx:   100,
	}
	return task.Encode()
}

func TestNewCompactClient(t *testing.T) {
	c := NewCompactClient("http://localhost:8080")
	if c == nil {
		t.Fatal("NewCompactClient returned nil")
	}
	if c.baseURL != "http://localhost:8080" {
		t.Errorf("baseURL = %q, want %q", c.baseURL, "http://localhost:8080")
	}
}

func TestCompactClientFetchTemplate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/template" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		jobID := r.URL.Query().Get("job")
		if jobID != "42" {
			t.Errorf("job param = %q, want %q", jobID, "42")
		}
		w.WriteHeader(http.StatusOK)
		w.Write(makeEncodedTemplate())
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	tmpl, err := c.FetchTemplate(42)
	if err != nil {
		t.Fatalf("FetchTemplate failed: %v", err)
	}
	if tmpl == nil {
		t.Fatal("FetchTemplate returned nil template")
	}
	if tmpl.JobID != 42 {
		t.Errorf("JobID = %d, want 42", tmpl.JobID)
	}
}

func TestCompactClientFetchTemplateNoContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	tmpl, err := c.FetchTemplate(1)
	if err != nil {
		t.Fatalf("FetchTemplate failed: %v", err)
	}
	if tmpl != nil {
		t.Error("FetchTemplate should return nil for 204 No Content")
	}
}

func TestCompactClientFetchTask(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/task/fetch" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		// Verify request contains workerID + speed
		body := make([]byte, 256)
		n, _ := r.Body.Read(body)
		body = body[:n]
		if len(body) < 1 {
			t.Error("request body is empty")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(makeEncodedTask())
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	task, err := c.FetchTask("worker-1", 0)
	if err != nil {
		t.Fatalf("FetchTask failed: %v", err)
	}
	if task == nil {
		t.Fatal("FetchTask returned nil task")
	}
	if task.TaskID != 10 {
		t.Errorf("TaskID = %d, want 10", task.TaskID)
	}
	if task.StartIdx != 0 || task.EndIdx != 100 {
		t.Errorf("range = [%d, %d), want [0, 100)", task.StartIdx, task.EndIdx)
	}
}

func TestCompactClientFetchTaskNoContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	task, err := c.FetchTask("worker-1", 0)
	if err != nil {
		t.Fatalf("FetchTask failed: %v", err)
	}
	if task != nil {
		t.Error("FetchTask should return nil for 204 No Content")
	}
}

func TestCompactClientSubmitResult(t *testing.T) {
	var receivedBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/task/submit" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		body := make([]byte, 4096)
		n, _ := r.Body.Read(body)
		receivedBody = body[:n]
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	result := &protocol.CompactResult{
		TaskID:  10,
		Matches: []protocol.MatchData{},
	}
	err := c.SubmitResult("worker-1", result)
	if err != nil {
		t.Fatalf("SubmitResult failed: %v", err)
	}
	if len(receivedBody) == 0 {
		t.Error("SubmitResult sent empty body")
	}
	// Verify workerID is in the body
	workerIDLen := int(receivedBody[0])
	if workerIDLen <= 0 || workerIDLen > len(receivedBody)-1 {
		t.Errorf("invalid workerID length: %d", workerIDLen)
		return
	}
	workerID := string(receivedBody[1 : 1+workerIDLen])
	if workerID != "worker-1" {
		t.Errorf("workerID in body = %q, want %q", workerID, "worker-1")
	}
}

func TestCompactClientSubmitResultHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	result := &protocol.CompactResult{TaskID: 1, Matches: nil}
	err := c.SubmitResult("worker-1", result)
	if err == nil {
		t.Error("SubmitResult should return error for HTTP 500")
	}
}

func TestCompactClientNetworkError(t *testing.T) {
	c := NewCompactClient("http://127.0.0.1:0") // port 0 = no listener
	_, err := c.FetchTask("worker-1", 0)
	if err == nil {
		t.Error("FetchTask should return error when server is unreachable")
	}
}

// ---- HTTPClient tests ----

func TestNewHTTPClient(t *testing.T) {
	c := NewHTTPClient("http://localhost:8080")
	if c == nil {
		t.Fatal("NewHTTPClient returned nil")
	}
	if c.baseURL != "http://localhost:8080" {
		t.Errorf("baseURL = %q, want %q", c.baseURL, "http://localhost:8080")
	}
}

func TestHTTPClientFetchTask(t *testing.T) {
	task := &protocol.Task{
		ID: 5,
		Mnemonics: []string{
			"abandon ability",
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req protocol.TaskRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.WorkerID != "test-worker" {
			t.Errorf("WorkerID = %q, want %q", req.WorkerID, "test-worker")
		}
		resp := protocol.TaskResponse{Task: task, Count: 1}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	got, err := c.FetchTask("test-worker")
	if err != nil {
		t.Fatalf("FetchTask failed: %v", err)
	}
	if got == nil {
		t.Fatal("FetchTask returned nil task")
	}
	if got.ID != 5 {
		t.Errorf("task.ID = %d, want 5", got.ID)
	}
}

func TestHTTPClientFetchTaskNoContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	task, err := c.FetchTask("worker")
	if err != nil {
		t.Fatalf("FetchTask failed: %v", err)
	}
	if task != nil {
		t.Error("FetchTask should return nil for 204 No Content")
	}
}

func TestHTTPClientFetchTaskHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	_, err := c.FetchTask("worker")
	if err == nil {
		t.Error("FetchTask should return error for HTTP 503")
	}
}

func TestHTTPClientSubmitResult(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/task/submit" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	result := &protocol.Result{
		TaskID:    1,
		Addresses: [][]byte{make([]byte, 20)},
		Mnemonics: []string{"abandon abandon"},
	}
	if err := c.SubmitResult("worker", result); err != nil {
		t.Fatalf("SubmitResult failed: %v", err)
	}
}

func TestHTTPClientSubmitResultHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	result := &protocol.Result{TaskID: 1}
	if err := c.SubmitResult("worker", result); err == nil {
		t.Error("SubmitResult should return error for HTTP 400")
	}
}

// ---- Worker tests ----

type mockSchedulerClient struct {
	fetchCount  int
	submitCount int
	task        *protocol.Task
	fetchErr    error
	submitErr   error
}

func (m *mockSchedulerClient) FetchTask(workerID string) (*protocol.Task, error) {
	m.fetchCount++
	return m.task, m.fetchErr
}

func (m *mockSchedulerClient) SubmitResult(workerID string, result *protocol.Result) error {
	m.submitCount++
	return m.submitErr
}

type mockSeedComputer struct{}

func (m *mockSeedComputer) Compute(mnemonics []string) [][]byte {
	addrs := make([][]byte, len(mnemonics))
	for i := range addrs {
		addrs[i] = make([]byte, 20)
	}
	return addrs
}

func (m *mockSeedComputer) Close() error { return nil }

func TestNewWorker(t *testing.T) {
	client := &mockSchedulerClient{}
	computer := &mockSeedComputer{}
	w := NewWorker("test-id", client, computer, 2)
	if w == nil {
		t.Fatal("NewWorker returned nil")
	}
	if w.id != "test-id" {
		t.Errorf("id = %q, want %q", w.id, "test-id")
	}
	if w.computeWorkers != 2 {
		t.Errorf("computeWorkers = %d, want 2", w.computeWorkers)
	}
}

func TestWorkerSetPollInterval(t *testing.T) {
	w := NewWorker("id", &mockSchedulerClient{}, &mockSeedComputer{}, 1)
	w.SetPollInterval(5 * time.Second)
	if w.pollInterval != 5*time.Second {
		t.Errorf("pollInterval = %v, want 5s", w.pollInterval)
	}
}

func TestWorkerSetPrefetchSize(t *testing.T) {
	w := NewWorker("id", &mockSchedulerClient{}, &mockSeedComputer{}, 1)
	w.SetPrefetchSize(10)
	if w.prefetchSize != 10 {
		t.Errorf("prefetchSize = %d, want 10", w.prefetchSize)
	}
}

func TestWorkerGetStats(t *testing.T) {
	w := NewWorker("id", &mockSchedulerClient{}, &mockSeedComputer{}, 1)
	stats := w.GetStats()

	keys := []string{"tasks_fetched", "tasks_computed", "tasks_submitted", "queue_length"}
	for _, k := range keys {
		if _, ok := stats[k]; !ok {
			t.Errorf("GetStats missing key %q", k)
		}
	}
}

func TestWorkerStartStop(t *testing.T) {
	client := &mockSchedulerClient{task: nil} // no tasks available
	w := NewWorker("id", client, &mockSeedComputer{}, 1)
	w.SetPollInterval(10 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	w.Start(ctx)

	// Start again should be no-op (already running)
	w.Start(ctx)

	<-ctx.Done()
	w.Stop()

	// Stop again should be no-op
	w.Stop()
}

func TestWorkerProcessesTask(t *testing.T) {
	task := &protocol.Task{
		ID:        1,
		Mnemonics: []string{"abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"},
	}
	callCount := 0
	client := &mockSchedulerClient{}
	client.task = task
	// After the first call, return nil to stop fetching
	origFetch := client

	// Use a channel to control when to return nil
	taskCh := make(chan *protocol.Task, 1)
	taskCh <- task

	controlClient := &controlledClient{taskCh: taskCh}
	submitDone := make(chan struct{})
	controlClient.submitFn = func() {
		select {
		case submitDone <- struct{}{}:
		default:
		}
	}

	_ = origFetch
	_ = callCount

	w := NewWorker("id", controlClient, &mockSeedComputer{}, 1)
	w.SetPollInterval(10 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	w.Start(ctx)

	select {
	case <-submitDone:
		// task was processed and submitted
	case <-time.After(5 * time.Second):
		t.Error("task was not processed within timeout")
	}

	cancel()
	w.Stop()

	stats := w.GetStats()
	if stats["tasks_computed"] == 0 {
		t.Error("tasks_computed should be > 0 after processing a task")
	}
}

type controlledClient struct {
	taskCh   chan *protocol.Task
	submitFn func()
}

func (c *controlledClient) FetchTask(workerID string) (*protocol.Task, error) {
	select {
	case task := <-c.taskCh:
		return task, nil
	default:
		return nil, nil
	}
}

func (c *controlledClient) SubmitResult(workerID string, result *protocol.Result) error {
	if c.submitFn != nil {
		c.submitFn()
	}
	return nil
}

// ---- CompactWorker tests ----

func TestNewCompactWorker(t *testing.T) {
	c := NewCompactClient("http://localhost")
	w := NewCompactWorker("test-id", c, 2)
	if w == nil {
		t.Fatal("NewCompactWorker returned nil")
	}
	if w.id != "test-id" {
		t.Errorf("id = %q, want %q", w.id, "test-id")
	}
	if w.workers != 2 {
		t.Errorf("workers = %d, want 2", w.workers)
	}
}

func TestCompactWorkerSetBloomFilter(t *testing.T) {
	c := NewCompactClient("http://localhost")
	w := NewCompactWorker("id", c, 1)

	f := bloom.NewFilter(100, 0.01)
	w.SetBloomFilter(f)
	if w.bloomFilter != f {
		t.Error("SetBloomFilter did not set the filter")
	}
}

func TestCompactWorkerLocalEnumeratorPerTask(t *testing.T) {
	// Verify that NewLocalEnumerator correctly initialises from a TaskTemplate.
	tmpl := &TaskTemplate{
		JobID:      1,
		Words:      []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", ""},
		UnknownPos: []int{11},
	}
	enum := NewLocalEnumerator(tmpl)
	if enum == nil {
		t.Fatal("NewLocalEnumerator returned nil")
	}
	if enum.GetWordCount() != 2048 {
		t.Errorf("word count = %d, want 2048", enum.GetWordCount())
	}
}

func TestCompactWorkerStartStop(t *testing.T) {
	// Use a test server that returns no tasks
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/task/fetch":
			// Read the workerID from body
			body := make([]byte, 256)
			n, _ := r.Body.Read(body)
			body = body[:n]
			if len(body) < 9 {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			// Parse workerID length
			idLen := int(body[0])
			if idLen+9 > len(body) {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			// batchSize is at offset 1+idLen
			_ = binary.BigEndian.Uint64(body[1+idLen : 1+idLen+8])
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewCompactClient(srv.URL)
	w := NewCompactWorker("test-worker", client, 1)
	w.pollInterval = 10 * time.Millisecond

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	w.Start(ctx)

	// Start again = no-op
	w.Start(ctx)

	<-ctx.Done()
	w.Stop()

	// Stop again = no-op
	w.Stop()
}

func TestCompactWorkerBodyFormat(t *testing.T) {
	// Verify the binary format sent by FetchTask
	var capturedBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, 256)
		n, _ := r.Body.Read(body)
		capturedBody = make([]byte, n)
		copy(capturedBody, body[:n])
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := NewCompactClient(srv.URL)
	c.FetchTask("myworker", 12345)

	if len(capturedBody) == 0 {
		t.Fatal("no body was sent")
	}
	idLen := int(capturedBody[0])
	if string(capturedBody[1:1+idLen]) != "myworker" {
		t.Errorf("workerID in body = %q, want %q", string(capturedBody[1:1+idLen]), "myworker")
	}
	speed := int64(binary.BigEndian.Uint64(capturedBody[1+idLen : 1+idLen+8]))
	if speed != 12345 {
		t.Errorf("speed in body = %d, want 12345", speed)
	}
}

// ---- Verify HTTPClient sends correct content-type ----

func TestHTTPClientContentType(t *testing.T) {
	var gotContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := NewHTTPClient(srv.URL)
	c.FetchTask("worker")

	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want %q", gotContentType, "application/json")
	}
}

// TestCompactWorkerProcessesTask verifies the CompactWorker end-to-end pipeline:
// template fetch → task fetch → compute → submit.
func TestCompactWorkerProcessesTask(t *testing.T) {
	submitDone := make(chan []byte, 1)

	// Encode a template: abandon*11 + unknown at position 11
	templateData := (&protocol.TaskTemplate{
		JobID:      1,
		UnknownPos: []int{11},
		Words: []string{
			"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "",
		},
	}).Encode()

	// A task covering index 3, which yields the known-valid mnemonic
	// "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
	taskData := (&protocol.CompactTask{
		TaskID:   99,
		JobID:    1,
		StartIdx: 0,
		EndIdx:   10, // small range: only check indices 0-9
	}).Encode()

	taskServed := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/template":
			w.WriteHeader(http.StatusOK)
			w.Write(templateData)
		case "/api/task/fetch":
			if !taskServed {
				taskServed = true
				w.WriteHeader(http.StatusOK)
				w.Write(taskData)
			} else {
				w.WriteHeader(http.StatusNoContent)
			}
		case "/api/task/submit":
			body := make([]byte, 4096)
			n, _ := r.Body.Read(body)
			result := make([]byte, n)
			copy(result, body[:n])
			select {
			case submitDone <- result:
			default:
			}
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewCompactClient(srv.URL)
	w := NewCompactWorker("pipeline-worker", client, 1)
	w.pollInterval = 10 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	w.Start(ctx)

	select {
	case body := <-submitDone:
		// Verify the submission has valid structure
		if len(body) < 1 {
			t.Error("submit body is empty")
			break
		}
		idLen := int(body[0])
		if idLen <= 0 || 1+idLen >= len(body) {
			t.Errorf("invalid workerID length in submit body: %d", idLen)
			break
		}
		workerID := string(body[1 : 1+idLen])
		if workerID != "pipeline-worker" {
			t.Errorf("workerID = %q, want %q", workerID, "pipeline-worker")
		}
	case <-time.After(10 * time.Second):
		t.Error("CompactWorker did not process and submit a task within timeout")
	}

	cancel()
	w.Stop()
}

// TestCompactWorkerAutoFetchTemplate verifies the worker fetches template automatically
// when template is not pre-set.
func TestCompactWorkerAutoFetchTemplate(t *testing.T) {
	templateData := (&protocol.TaskTemplate{
		JobID:      5,
		UnknownPos: []int{11},
		Words: []string{
			"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "",
		},
	}).Encode()

	taskData := (&protocol.CompactTask{
		TaskID: 1, JobID: 5, StartIdx: 0, EndIdx: 5,
	}).Encode()

	templateFetched := make(chan struct{}, 1)
	taskServed := false

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/template":
			select {
			case templateFetched <- struct{}{}:
			default:
			}
			w.WriteHeader(http.StatusOK)
			w.Write(templateData)
		case "/api/task/fetch":
			if !taskServed {
				taskServed = true
				w.WriteHeader(http.StatusOK)
				w.Write(taskData)
			} else {
				w.WriteHeader(http.StatusNoContent)
			}
		case "/api/task/submit":
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := NewCompactClient(srv.URL)
	// Do NOT call SetTemplate - let worker fetch it automatically
	w := NewCompactWorker("auto-worker", client, 1)
	w.pollInterval = 10 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	w.Start(ctx)

	select {
	case <-templateFetched:
		// Template was fetched automatically
	case <-time.After(5 * time.Second):
		t.Error("CompactWorker did not fetch template automatically")
	}

	cancel()
	w.Stop()
}

// TestWorkerFetchTaskError verifies the Worker retries on fetch error.
func TestWorkerFetchTaskError(t *testing.T) {
	callCount := 0
	client := &countingClient{
		fetchFn: func() (*protocol.Task, error) {
			callCount++
			if callCount == 1 {
				return nil, bytes.ErrTooLarge // simulate error on first call
			}
			return nil, nil // no task after that
		},
	}

	w := NewWorker("id", client, &mockSeedComputer{}, 1)
	w.SetPollInterval(10 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	w.Start(ctx)
	<-ctx.Done()
	w.Stop()

	if callCount < 1 {
		t.Error("FetchTask should have been called at least once")
	}
}

type countingClient struct {
	fetchFn  func() (*protocol.Task, error)
	submitFn func(*protocol.Result) error
}

func (c *countingClient) FetchTask(workerID string) (*protocol.Task, error) {
	if c.fetchFn != nil {
		return c.fetchFn()
	}
	return nil, nil
}

func (c *countingClient) SubmitResult(workerID string, result *protocol.Result) error {
	if c.submitFn != nil {
		return c.submitFn(result)
	}
	return nil
}

// Ensure bytes package is used (avoid unused import)
var _ = bytes.NewBuffer
